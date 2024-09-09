from typing import Callable, Optional

import torch.nn as nn

from torch import Tensor

from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from models.nn.attention import AAFTAttention, GTAttention
from rl4co.models.nn.ops import Normalization, SkipConnection
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class GTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = True,
        gt_type: str = "gt",
        moe_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")
        
        if gt_type == "gt":
            self.gt = GTAttention(embed_dim=embed_dim, n_heads=n_heads, bias=bias)
        elif gt_type == "aaft":
            self.gt = AAFTAttention(embed_dim=embed_dim, bias=bias)
        
        self.post_gt = nn.Sequential(
            Normalization(embed_dim, normalization),
            SkipConnection(ffn),
            Normalization(embed_dim, normalization),
        )

    def forward(self, x, dist):
        x = x + self.gt(x, dist)
        x = self.post_gt(x)
        return x


class GTNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        n_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        gt_type: str = "gt",
        moe_kwargs: Optional[dict] = None,
    ):
        super(GTNetwork, self).__init__()

        self.layers = nn.ModuleList(
            (
                GTLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    gt_type = gt_type,
                    moe_kwargs=moe_kwargs,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: Tensor, dist: Tensor) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            dist: [batch_size, graph_size, graph_size]
        """
        for layer in self.layers:
            x = layer(x, dist)
        return x
