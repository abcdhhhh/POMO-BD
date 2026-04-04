import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.attention import MultiHeadAttention

class GTAttention(MultiHeadAttention):
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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, bias=bias, **factory_kwargs)
        self.We = nn.Linear(1, num_heads, bias=bias, **factory_kwargs)
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor, attn_mask=None):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        e = rearrange(
            self.We(dist.unsqueeze(-1)), "b s1 s2 h -> b h s1 s2"
        )
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                e.masked_fill_(~attn_mask, float("-inf"))
            else:
                e += attn_mask
        
        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=e,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)")) 