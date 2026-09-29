from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def sinkhorn_normalization(log_alpha: Tensor, n_iters: int = 10) -> Tensor:
    """Applies Sinkhorn normalization to a log_alpha tensor.

    Args:
        log_alpha: [..., n, n] unnormalized log probabilities
        n_iters: number of normalization iterations

    Returns:
        log_alpha: [..., n, n] doubly normalized log probabilities
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha

def scaled_dot_product_attention_sinkhorn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Simple (exact) Scaled Dot-Product Attention in RL4CO without customized kernels (i.e. no Flash Attention)."""

    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

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

    # Apply Sinkhorn normalization
    scores = sinkhorn_normalization(scores)

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)