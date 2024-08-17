import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class AAFT(nn.Module):
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
    
    def forward(self, x, dist):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three d) -> three b s d", three=3
        ).unbind(dim=0)
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=1, keepdim=True)[0])
        w = -self.alpha * dist
        w = torch.exp(w - torch.max(w, dim=2, keepdim=True)[0])
        wkv = torch.einsum('bij, bjd -> bid', w, k*v)
        wk = torch.einsum('bij, bjd -> bid', w, k)
        out = q * wkv / (wk + 1e-8)

        assert not torch.isnan(out).any()
        
        return self.out_proj(out)
        
    