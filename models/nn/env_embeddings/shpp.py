import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext

class SHPPInitEmbedding(nn.Module):
    """Initial embedding for the SHPP.
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SHPPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_endpoint = nn.Linear(node_dim, embed_dim, linear_bias)  # endpoint embedding, may swap

    def forward(self, td):
        first, way, last = td["locs"][:, :1, :], td["locs"][:, 1:-1, :], td["locs"][:, -1:, :]
        first_embedding = self.init_embed_endpoint(first)
        last_embedding = self.init_embed_endpoint(last)
        node_embeddings = self.init_embed(way)
        out = torch.cat((first_embedding, node_embeddings, last_embedding), -2)
        return out
    
class SHPPContext(EnvContext):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - last node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(SHPPContext, self).__init__(embed_dim, embed_dim * 2)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embed_dim * 2).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["last_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)
        