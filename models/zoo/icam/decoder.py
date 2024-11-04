from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from rl4co.utils.ops import batchify, unbatchify, gather_by_index
import math

log = get_pylogger(__name__)

@dataclass
class BiasCache(PrecomputedCache):
    bias: Tensor

class ICAMDecoder(AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        encoder_type: str = "aft",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
        pointer: Union[nn.Module, str] = None,
        moe_kwargs: dict = None,
    ):
        self.encoder_type = encoder_type
        self.pointer_type = pointer
        if pointer == "aft":
            from models.nn.attention import AFTPointerAttention
            pointer = AFTPointerAttention(
                mask_inner=mask_inner,
                check_nan=check_nan,
            )
        elif pointer == "poly":
            from rl4co.models.nn.attention import PolyNetAttention
            pointer = PolyNetAttention(
                k=128,
                embed_dim=embed_dim,
                poly_layer_dim=embed_dim*2,
                num_heads=num_heads,
                mask_inner=mask_inner,
                out_bias=out_bias_pointer_attn,
                check_nan=check_nan,
            )

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
            pointer=pointer,
            moe_kwargs=moe_kwargs,
        )
        if self.pointer_type == "aft":
            self.alpha_aft = nn.Parameter(torch.tensor(1.))
        self.alpha = nn.Parameter(torch.tensor(1.))
        

    def pre_decoder_hook(
        self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, BiasCache]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, self._precompute_cache(td, embeddings, num_starts=num_starts)
    
    def _precompute_cache(
        self, td, embeddings: Tensor, num_starts: int = 0
    ) -> BiasCache:
        if self.encoder_type == "aft":
            return self._precompute_cache_aft(td, embeddings, num_starts)
        elif self.encoder_type == "mataft":
            return self._precompute_cache_mataft(td, embeddings, num_starts)

    def _precompute_cache_aft(
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
        dist = torch.cdist(locs, locs, p=2)
        bias = self.alpha * math.log2(n) * dist
        if self.pointer_type == "aft":
            bias_aft = self.alpha_aft * math.log2(n) * dist
            bias_aft = torch.exp(-bias_aft)
            bias = torch.stack([bias_aft, bias], dim=0)

        # Organize in a dataclass for easy access
        return BiasCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
            bias=bias,
        )
    
    def _precompute_cache_mataft(
        self, td, embeddings: torch.Tensor, num_starts: int = 0
    ) -> BiasCache:
        col_emb, row_emb = embeddings
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key,
        ) = self.project_node_embeddings(
            col_emb
        ).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(col_emb.mean(1))
        else:
            graph_context = 0

        dmat = td["cost_matrix"]
        if num_starts > 1:
            dmat = unbatchify(dmat, num_starts)
            assert (dmat == dmat[:, [0]]).all()
            dmat = dmat[:, 0]
        n = dmat.size(-1)
        bias = self.alpha * math.log2(n) * dmat
        if self.pointer_type == "aft":
            bias_aft = self.alpha_aft * math.log2(n) * dmat
            bias_aft = torch.exp(-bias_aft)
            bias = torch.stack([bias_aft, bias], dim=0)

        # Organize in a dataclass for easy access
        return BiasCache(
            node_embeddings=row_emb,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key,
            bias=bias
        )

    def forward(
        self,
        td: TensorDict,
        cached: BiasCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings and bias
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

        current_node = td["current_node"]
        if (num_starts > 1 and current_node.ndim == 3) or (num_starts == 1 and current_node.ndim == 2):
            current_node = current_node.squeeze(-1)
        
        # Compute logits
        mask = td["action_mask"]
        if self.pointer_type == "aft":
            bias_aft = gather_by_index(cached.bias[0], current_node)
            logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, bias_aft, mask)
            bias = gather_by_index(cached.bias[1], current_node)
        else:
            logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)
            bias = gather_by_index(cached.bias, current_node)

        if self.env_name in ["tsp", "atsp"] and (td["i"] == 0).any():
            pass # not connected
        else:
            logits = logits - bias
        
        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)
        return logits, mask
