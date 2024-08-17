import torch
import torch.nn as nn

class SHPPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SHPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_start = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_end = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        start_embed = self.init_embed_start(td["locs"][:, :1])
        node_embed = self.init_embed(td["locs"][:, 1:-1])
        end_embed = self.init_embed_end(td["locs"][:, -1:])
        out = torch.cat([start_embed, node_embed, end_embed], dim=-2)
        return out