import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.attention import PointerAttention
import math

class GTAttention(nn.Module):
    """ 
    following MARCO (https://arxiv.org/abs/2408.02207) and GT (https://arxiv.org/abs/2012.09699)
    inputs include a distance matrix
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.We = nn.Linear(1, num_heads, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor, attn_mask=None):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads,
        ).unbind(dim=0)
        e = rearrange(
            self.We(dist.unsqueeze(-1)), "b s1 s2 h -> b h s1 s2"
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        scores = scores + e
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(~attn_mask, float("-inf"))
            else:
                scores += attn_mask
        scores = F.softmax(scores, dim=-1)

        out = torch.matmul(scores, v)
        
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))      