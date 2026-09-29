import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ...nn.ops import SwiGLU, Normalization

from rl4co.models.nn.attention import PointerAttention
import math

class ReLDPointerAttention(PointerAttention):
    """
    follow ReLD (http://arxiv.org/abs/2503.00753)
    adding idt and ffn
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        **kwargs,
    ):
        # embed_dim, num_heads is not needed
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, **kwargs)
        hidden_dim = embed_dim * 8 // 3
        # self.norm = Normalization(embed_dim, "rms")
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 2),
            SwiGLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, cur_node_embedding, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            cur_node_embedding: current node embedding tensor of shape [B, ..., L, E]
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = heads # + cur_node_embedding
        # glimpse = self._project_out(heads, attn_mask)
        # glimpse = self.ffn(glimpse) + glimpse

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

