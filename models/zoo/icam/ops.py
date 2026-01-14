import math

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.utils.ops import gather_by_index

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": nn.RMSNorm,
            }.get(normalization, None)
            
            if normalization == "rms":
                self.normalizer = normalizer_class(embed_dim)
            else:
                self.normalizer = normalizer_class(embed_dim, affine=True)
        else:
            self.normalizer = "layer"

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.RMSNorm):
            return self.normalizer(x)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x
        
class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden, normalization="batch", activation="relu") -> None:
        super().__init__()

        if activation == "relu":
            ffn = nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
        elif activation == "swiglu":
            feed_forward_hidden = feed_forward_hidden * 2 // 3
            ffn = nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden * 2),
                SwiGLU(),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(embed_dim, normalization),
                "ffn": ffn,
                "norm2": Normalization(embed_dim, normalization),
            }
        )

    def forward(self, x, x_old):
        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x
    
    
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return x
    
if __name__ == "__main__":
    # simple test
    x = torch.randn(2, 4, 8)
    ffn = TransformerFFN(embed_dim=8, feed_forward_hidden=32, normalization="rms", activation="swiglu")
    y = ffn(x, x)
    print(y.shape)  # should be [2, 4, 8]