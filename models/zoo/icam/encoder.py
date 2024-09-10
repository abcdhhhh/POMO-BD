from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from models.nn.graph.attnnet import GTNetwork


class ICAMEncoder(AutoregressiveEncoder):
    """
    following ICAM (https://arxiv.org/abs/2405.01906)
    NOTE: precision 32-true is recommended, may get NaN if precision is low
    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        gt_type: str = "aaft",
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        moe_kwargs: dict = None,
    ):
        super(ICAMEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        self.net = (
            GTNetwork(
                num_layers = num_layers,
                embed_dim = embed_dim,
                num_heads = num_heads,
                normalization = normalization,
                feedforward_hidden = feedforward_hidden,
                gt_type = gt_type,
                moe_kwargs = moe_kwargs,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        locs = td["locs"]
        dist = torch.cdist(locs, locs, p=2)

        # Process embedding
        h = self.net(init_h, dist)

        # Return latent representation and initial embedding
        return h, init_h
