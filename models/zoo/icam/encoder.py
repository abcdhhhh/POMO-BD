from typing import Tuple, Union

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from models.nn.graph.attnnet import AFTNetwork, MatAFTNetwork


class AFTEncoder(AutoregressiveEncoder):
    """
    following ICAM (https://arxiv.org/abs/2405.01906)
    NOTE: precision 32-true is recommended, may get NaN if precision is low
    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_layers: int = 3,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        self.net = (
            AFTNetwork(
                num_layers = num_layers,
                embed_dim = embed_dim,
                normalization = normalization,
                feedforward_hidden = feedforward_hidden,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, attn_mask: Tensor = None
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
        h = self.net(init_h, dist, attn_mask)

        # Return latent representation and initial embedding
        return h, init_h

class MatAFTEncoder(AutoregressiveEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        env_name: str = "atsp",
        num_layers: int = 3,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        net: nn.Module = None,
    ):
        super().__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        if init_embedding is None:
            init_embedding = MatAFTInitEmbedding(
                embed_dim=embed_dim,
                **init_embedding_kwargs,
            )
            # init_embedding = env_init_embedding(
            #     "matnet", {"embed_dim": embed_dim, **init_embedding_kwargs}
            # )
        self.init_embedding = init_embedding
        self.net = (
            MatAFTNetwork(
                num_layers = num_layers,
                embed_dim = embed_dim,
                normalization = normalization,
                feedforward_hidden = feedforward_hidden,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, attn_mask: Tensor = None
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
        row_emb, col_emb, dmat = self.init_embedding(td)

        # Process embedding
        row_emb, col_emb = self.net(row_emb, col_emb, dmat, attn_mask)

        # Return latent representation and initial embedding
        return (row_emb, col_emb), None


class MatAFTInitEmbedding(nn.Module):

    def __init__(self, embed_dim: int, linear_bias=True, topk: int = 50) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.topk = topk
        self.row_init_embed = nn.Linear(topk, embed_dim, linear_bias)
        self.col_init_embed = nn.Linear(topk, embed_dim, linear_bias)

    def forward(self, td: TensorDict):
        dmat = td["cost_matrix"]

        row_topk = dmat.topk(k=self.topk, dim=-1, largest=False, sorted=True)[0]
        row_emb = self.row_init_embed(row_topk)

        col_topk = dmat.transpose(-2, -1).topk(k=self.topk, dim=-1, largest=False, sorted=True)[0]
        col_emb = self.col_init_embed(col_topk)

        return row_emb, col_emb, dmat