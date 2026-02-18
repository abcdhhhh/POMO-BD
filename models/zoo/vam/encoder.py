from rl4co.models.zoo.am.encoder import AttentionModelEncoder
import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase
from rl4co.models.common.constructive import AutoregressiveEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.models.nn.graph.attnnet import GraphAttentionNetwork


class VAMEncoder(AttentionModelEncoder):
    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "tsp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn=None,
        moe_kwargs: dict = None,
    ):
        super().__init__()
        self.global_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, td: TensorDict, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        init_h = self.init_embedding(td)

        init_h = torch.cat([init_h, self.global_embedding.expand_as(init_h[:, :1, :])], dim=1)
        assert mask is None

        # Process embedding
        h = self.net(init_h, mask)

        # Return latent representation and initial embedding
        return h, init_h