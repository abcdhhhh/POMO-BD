import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def scaled_dot_product_graph_attention_simple(
    q, k, v, e1, e2, attn_mask=None, dropout_p=0.0, is_causal=False
):
    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    scores = scores + e1

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = attn_weights * e2

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)

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
        attention_dropout: float = 0.0,
        causal: bool = False,
        device: str = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        self.attention_dropout = attention_dropout
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        factory_kwargs = {"device": device, "dtype": dtype}
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.We = nn.Linear(1, 2 * num_heads, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor, attn_mask=None):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads,
        ).unbind(dim=0)
        e1, e2 = rearrange(
            self.We(dist.unsqueeze(-1)), "b i j (two h) -> two b h i j", two=2
        ).unbind(dim=0)

        out = scaled_dot_product_graph_attention_simple(
            q, k, v, e1, e2, attn_mask=attn_mask, dropout_p=self.attention_dropout, is_causal=self.causal,
        )

        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))  


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
        self.beta = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor):
        """
        x: (batch, seqlen, hidden_dim)
        dist: (batch, seqlen, seqlen)
        """
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three d) -> three b s d", three=3,
        ).unbind(dim=0)
        q = F.sigmoid(q)
        k = torch.exp(k - torch.max(k, dim=-2, keepdim=True)[0])
        w = -self.alpha * torch.tensor(x.size(1)).log2() * dist + self.beta * torch.eye(x.size(1), device=x.device)
        w = torch.exp(w - torch.max(w, dim=-1, keepdim=True)[0])
        wkv = torch.einsum('b i j, b j d -> b i d', w, k*v)
        wk = torch.einsum('b i j, b j d -> b i d', w, k)
        wk = wk + (1e-10 if wk.dtype != torch.float16 else 3e-8)
        out = q * wkv / wk
        assert not torch.isnan(out).any()
        
        return self.out_proj(out)
