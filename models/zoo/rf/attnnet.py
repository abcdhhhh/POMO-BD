from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from rl4co.models.nn.attention import MultiHeadAttention
from ...nn.ops import TransformerFFN, Normalization
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class TELayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "rms",
        activation: str = "swiglu",
        bias: bool = False,
        sdpa_fn: Optional[Callable] = None,
    ):
        super().__init__()  
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, bias=bias, sdpa_fn=sdpa_fn,
        )
        self.ffn = TransformerFFN(embed_dim, feedforward_hidden, normalization, activation)

    def forward(self, x, attn_mask=None):
        x_norm = self.ffn.ops["norm1"](x)
        x_out = self.mha(x_norm, attn_mask=attn_mask)
        x = x + x_out
        x_norm = self.ffn.ops["norm2"](x)
        x_out = self.ffn.ops["ffn"](x_norm)
        x = x + x_out
        return x
    
class TENetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        normalization: str = "rms",
        activation: str = "swiglu",
        bias: bool = False,
        sdpa_fn: Optional[Callable] = None,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                TELayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    activation=activation,
                    bias=bias,
                    sdpa_fn=sdpa_fn,
                )
                for _ in range(num_layers)
            )
        )
        self.post_norm = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, attn_mask: Tensor=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            attn_mask: [batch_size, ..., graph_size]
        """
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.post_norm(x)
        return x