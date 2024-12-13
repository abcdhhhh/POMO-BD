from typing import Callable, Optional

import torch.nn as nn

from torch import Tensor

from models.zoo.icam.attention import AFTAttention, AFTCrossAttention
from rl4co.models.nn.ops import TransformerFFN
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AFTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = False,
    ):
        super().__init__()  
        self.aft = AFTAttention(embed_dim=embed_dim, bias=bias)
        self.ffn = TransformerFFN(embed_dim, feedforward_hidden, normalization)

    def forward(self, x, dist, attn_mask=None):
        x_out = x + self.aft(x, dist, attn_mask)
        x = self.ffn(x_out, x)
        return x

class AFTNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                AFTLayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: Tensor, dist: Tensor, attn_mask: Tensor=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            dist: [batch_size, graph_size, graph_size]
            attn_mask: [batch_size, ..., graph_size]
        """
        for layer in self.layers:
            x = layer(x, dist, attn_mask)
        return x


class MatAFTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        bias: bool = False,
    ):
        super().__init__()
        self.row_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias)
        self.col_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias)
        
        self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
        self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)


    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        row_emb_out = self.row_aft(
            row_emb, col_emb, dmat=dmat, cross_attn_mask=attn_mask
        )
        attn_mask_t = attn_mask.transpose(-2, -1) if attn_mask is not None else None
        col_emb_out = self.col_aft(
            col_emb,
            row_emb,
            dmat=dmat.transpose(-2, -1),
            cross_attn_mask=attn_mask_t,
        )

        row_emb_out = self.F_a(row_emb_out, row_emb)
        col_emb_out = self.F_b(col_emb_out, col_emb)

        return row_emb_out, col_emb_out
    
class MatAFTNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                MatAFTLayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, row_emb: Tensor, col_emb: Tensor, dmat: Tensor, attn_mask: Tensor=None):
        """Forward pass of the encoder

        Args:
            row_emb: [b, m, d]
            col_emb: [b, n, d]
            dmat: [b, m, n]
            attn_mask: [b, ..., n]
        """
        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, dmat, attn_mask)
        return row_emb, col_emb
