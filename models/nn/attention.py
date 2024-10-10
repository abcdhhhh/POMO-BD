import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from rl4co.models.nn.attention import PointerAttention
import math

class AFTAttention(nn.Module):
    """ 
    following ICAM (https://arxiv.org/abs/2405.01906) and AFT-full (https://arxiv.org/abs/2105.14103)
    inputs include a distance matrix
    memory O(bn^2+bnhd)
    NOTE: may get NaN when training with 16-mixed
    """
    def __init__(
        self,
        embed_dim: int,
        bias: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.beta = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor, attn_mask=None):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three d) -> three b s d", three=3,
        ).unbind(dim=0)
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=-2, keepdim=True)[0])
        w = -self.alpha * math.log2(x.size(-2)) * dist - self.beta * torch.eye(x.size(-2), device=x.device)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                w.masked_fill_(~attn_mask, float("-inf"))
            else:
                w += attn_mask
        w = torch.exp(w)
        wkv = torch.bmm(w, k*v)
        wk = torch.bmm(w, k)
        out = q * (wkv / wk)
        
        return out
    
class AFTCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bias: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        self.alpha = nn.Parameter(torch.tensor(1.))
    
    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor, dmat: torch.Tensor, cross_attn_mask=None):
        """
        q_input: (batch, m, hidden_dim)
        kv_input: (batch, n, hidden_dim)
        dmat: (batch, m, n)
        cross_attn_mask: (batch, ..., n)
        """
        q = self.Wq(q_input)
        k, v = rearrange(
            self.Wkv(kv_input), "b n (two d) -> two b n d", two=2,
        ).unbind(dim=0)
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=-2, keepdim=True)[0])
        w = -self.alpha * math.log2(k.size(-2)) * dmat
        if cross_attn_mask is not None:
            if cross_attn_mask.ndim == 2:
                cross_attn_mask = cross_attn_mask.unsqueeze(1)
            if cross_attn_mask.dtype == torch.bool:
                w.masked_fill_(~cross_attn_mask, float("-inf"))
            else:
                w += cross_attn_mask
        w = torch.exp(w)
        wkv = torch.bmm(w, k*v)
        wk = torch.bmm(w, k)
        out = q * (wkv / wk)
        
        return out

class AFTPointerAttention(PointerAttention):
    def __init__(
        self,
        mask_inner: bool = True,
        check_nan: bool = True,
        **kwargs,
    ):
        super(PointerAttention, self).__init__()
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.check_nan = check_nan

    def forward(self, q, k, v, logit_key, w, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            w: [B, ..., S]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=-2, keepdim=True)[0])
        # Apply the provided attention mask
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = (attn_mask == -torch.inf)
            w.masked_fill_(~attn_mask, 0.)
        if w.ndim == 2:
            w = w.unsqueeze(1)
        w = torch.exp(w)
        wkv = torch.bmm(w, k*v)
        wk = torch.bmm(w, k)
        glimpse = q * (wkv / wk)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))
        
        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"
        return logits
