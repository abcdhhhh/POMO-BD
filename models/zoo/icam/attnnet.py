from typing import Callable, Optional

import torch
import torch.nn as nn

from torch import Tensor

from models.zoo.icam.attention import AFTAttention, AFTCrossAttention
from ...nn.ops import TransformerFFN, Normalization
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class AFTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        activation: str = "relu",
        idt: bool = False,
        bias: bool = False,
    ):
        super().__init__()  
        self.aft = AFTAttention(embed_dim=embed_dim, bias=bias, idt=idt)
        self.ffn = TransformerFFN(embed_dim, feedforward_hidden, True, normalization, activation)

    def forward(self, x, dist, attn_mask=None):
        x_out = self.aft(x, dist, attn_mask)
        x = self.ffn(x_out, x)
        return x

class AFTNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        normalization: str = "batch",
        activation: str = "relu",
        idt: bool = False,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                AFTLayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    activation=activation,
                    idt=idt,
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
    
class AFTELayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "rms",
        activation: str = "swiglu",
        idt: bool = False,
        bias: bool = False,
    ):
        super().__init__()  
        self.aft = AFTAttention(embed_dim=embed_dim, bias=bias, idt=idt)
        self.ffn = TransformerFFN(embed_dim, feedforward_hidden, bias, normalization, activation)

    def forward(self, x, dist, attn_mask=None):
        x_norm = self.ffn.ops["norm1"](x)
        x_out = self.aft(x_norm, dist, attn_mask=attn_mask)
        x = x + x_out
        x_norm = self.ffn.ops["norm2"](x)
        x_out = self.ffn.ops["ffn"](x_norm)
        x = x + x_out
        return x
    
class AFTENetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        normalization: str = "rms",
        activation: str = "swiglu",
        idt: bool = False,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                AFTELayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    activation=activation,
                    idt=idt,
                )
                for _ in range(num_layers)
            )
        )
        self.post_norm = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, dist: Tensor, attn_mask: Tensor=None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            attn_mask: [batch_size, ..., graph_size]
        """
        for layer in self.layers:
            x = layer(x, dist, attn_mask)
        x = self.post_norm(x)
        return x


class MatAFTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        activation: str = "relu",
        idt: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.row_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias, idt=idt)
        self.col_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias, idt=idt)

        self.F_a = TransformerFFN(embed_dim, feedforward_hidden, True, normalization, activation)
        self.F_b = TransformerFFN(embed_dim, feedforward_hidden, True, normalization, activation)


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
        activation: str = "relu",
        idt: bool = False,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                MatAFTLayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    activation=activation,
                    idt=idt,
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


class HeterAFTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        activation: str = "relu",
        idt: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.all_aft = AFTAttention(embed_dim=embed_dim, bias=bias, idt=idt)
        self.d2p_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias, idt=True)
        self.p2d_aft = AFTCrossAttention(embed_dim=embed_dim, bias=bias, idt=True)
        self.ffn = TransformerFFN(embed_dim, feedforward_hidden, True, normalization, activation)


    def forward(self, x, dist, attn_mask=None):

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = x.size()

        # Check if graph size is odd number
        assert (
            graph_size % 2 == 1
        ), "Graph size should have odd number of nodes due to pickup-delivery problem  \
                                     (n/2 pickup, n/2 delivery, 1 depot)"
        n_pick = (graph_size - 1) // 2
        # pickup -> its delivery
        x_p = (
            x[:, 1:n_pick+1, :]
        )  # [batch_size * n_pick, embed_dim]
        x_d = (
            x[:, n_pick+1:, :]
        )  # [batch_size * n_pick, embed_dim]

        dist_pd = dist[:, 1:n_pick+1, n_pick+1:]
        dist_dp = dist[:, n_pick+1:, 1:n_pick+1]

        depot_out = torch.zeros_like(x[:, :1, :])
        p_out = self.d2p_aft(x_p, x_d, dist_pd)
        d_out = self.p2d_aft(x_d, x_p, dist_dp)
        all_out = self.all_aft(x, dist, attn_mask)
        x_out = all_out + torch.cat([depot_out, p_out, d_out], dim=1)
        x = self.ffn(x_out, x)
        return x

class HeterAFTNetwork(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        normalization: str = "batch",
        activation: str = "relu",
        idt: bool = False,
        feedforward_hidden: int = 512,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            (
                HeterAFTLayer(
                    embed_dim=embed_dim,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    activation=activation,
                    idt=idt,
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