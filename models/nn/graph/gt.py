import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class GTLayer(nn.Module):
    """
    Graph Transformer
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: str = "batch",
        bias: bool = True,
        moe_kwargs: dict = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.W_h = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.W_e = nn.Linear(embed_dim, 2 * num_heads, bias=bias)

    def forward(self, h, e):
        """
        h [B, N, E]
        e [B, N, N, 1]
        """
        q, k, v = rearrange(
            self.W_h(h), "b s (three d) -> three b s d", three=3
        ).unbind(dim=0)
