from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn

from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import batchify, unbatchify

log = get_pylogger(__name__)

@dataclass
class BiasCache(PrecomputedCache):
    bias: Tensor

class ICAMDecoder(AttentionModelDecoder):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(1.))

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, BiasCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(td, embeddings, num_starts=num_starts)

    def _precompute_cache(
        self, td, embeddings: torch.Tensor, num_starts: int = 0
    ) -> BiasCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        locs = td["locs"]
        if num_starts > 1:
            locs = unbatchify(locs, num_starts)
            assert (locs == locs[:, [0]]).all()
            locs = locs[:, 0]
        n = locs.size(-2)
        bias = self.alpha * torch.cdist(locs, locs, p=2) * torch.tensor(n).log2()

        # Organize in a dataclass for easy access
        return BiasCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
            bias=bias,
        )

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Compute logits
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        n = cached.bias.size(-1)
        b = cached.bias.size(0)
        current_node = td["current_node"]
        if num_starts > 1:
            current_node = current_node[..., None].expand(*current_node.size(), n)
            bias = torch.gather(cached.bias, dim=-2, index=current_node)
        else:
            bias = cached.bias[torch.arange(b), current_node]
        logits = logits - bias

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask
