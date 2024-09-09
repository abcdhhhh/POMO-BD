import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class AAFTAttention(nn.Module):
    """ 
    following ICAM (https://arxiv.org/abs/2405.01906) and AFT-full (https://arxiv.org/abs/2105.14103)
    inputs include a distance matrix
    """
    def __init__(
        self,
        embed_dim: int,
        bias: bool = True,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.alpha = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three d) -> three b s d", three=3
        ).unbind(dim=0)
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=1, keepdim=True)[0])
        w = -self.alpha * dist * torch.tensor(x.size(1)).log2()
        w = torch.exp(w - torch.max(w, dim=2, keepdim=True)[0])
        wkv = torch.einsum('bij, bjd -> bid', w, k*v)
        wk = torch.einsum('bij, bjd -> bid', w, k)
        out = q * wkv / (wk + 1e-8)

        assert not torch.isnan(out).any()
        
        return self.out_proj(out)
        
class GTAttention(nn.Module):
    """ 
    following MARCO (https://arxiv.org/abs/2408.02207) and GT (https://arxiv.org/abs/2012.09699)
    inputs include a distance matrix
    """
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        bias: bool = True,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.We = nn.Linear(1, 2 * n_heads, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.n_heads,
        ).unbind(dim=0)
        e1, e2 = rearrange(
            self.We(dist.unsqueeze(-1)), "b s1 s2 (two h) -> two b h s1 s2", two=2
        ).unbind(dim=0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        scores = scores + e1
        scores = F.softmax(scores, dim=-1)
        scores = scores * e2

        out = torch.matmul(scores, v)
        
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))      
