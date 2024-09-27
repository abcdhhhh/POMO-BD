from typing import Optional, Union

import torch.nn as nn
from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveDecoder,
    NonAutoregressiveEncoder,
    NonAutoregressivePolicy,
)
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

from rl4co.models.zoo.nargnn import NARGNNEncoder

log = get_pylogger(__name__)


class DGNNPolicy(NonAutoregressivePolicy):
    """
    TODO: Dynamically re-embed after a few actions
    """

    def __init__(
        self,
        encoder: Optional[NonAutoregressiveEncoder] = None,
        decoder: Optional[NonAutoregressiveDecoder] = None,
        embed_dim: int = 64,
        env_name: str = "tsp",
        init_embedding: Optional[nn.Module] = None,
        edge_embedding: Optional[nn.Module] = None,
        graph_network: Optional[nn.Module] = None,
        heatmap_generator: Optional[nn.Module] = None,
        num_layers_heatmap_generator: int = 3,
        num_layers_graph_encoder: int = 12,
        act_fn="silu",
        agg_fn="mean",
        linear_bias: bool = True,
        train_decode_type: str = "multistart_sampling",
        val_decode_type: str = "multistart_greedy",
        test_decode_type: str = "multistart_greedy",
        k_sparse: int = 100,
        **constructive_policy_kw,
    ):
        if len(constructive_policy_kw) > 0:
            log.warning(f"Unused kwargs: {constructive_policy_kw}")

        if encoder is None:
            encoder = NARGNNEncoder(
                embed_dim=embed_dim,
                env_name=env_name,
                init_embedding=init_embedding,
                edge_embedding=edge_embedding,
                graph_network=graph_network,
                heatmap_generator=heatmap_generator,
                num_layers_heatmap_generator=num_layers_heatmap_generator,
                num_layers_graph_encoder=num_layers_graph_encoder,
                act_fn=act_fn,
                agg_fn=agg_fn,
                linear_bias=linear_bias,
                k_sparse=k_sparse,
            )

        # The decoder generates logits given the current td and heatmap
        if decoder is None:
            decoder = NonAutoregressiveDecoder()
        else:
            # check if the decoder has trainable parameters
            if any(p.requires_grad for p in decoder.parameters()):
                log.error(
                    "The decoder contains trainable parameters. This should not happen in a non-autoregressive policy."
                )

        # Pass to constructive policy
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **constructive_policy_kw,
        )