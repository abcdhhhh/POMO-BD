import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext

class SymPCTSPInitEmbedding(nn.Module):
    """Initial embedding for the Prize Collecting Traveling Salesman Problems (PCTSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot and customers separately)
        - expected_prize: expected prize for visiting the customers.
            In PCTSP, this is the actual prize. In SPCTSP, this is the expected prize.
        - penalty: penalty for not visiting the customers
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SymPCTSPInitEmbedding, self).__init__()
        node_dim = 4  # x, y, prize, penalty
        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)

    def forward(self, td):
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        node_embeddings = self.init_embed(
            torch.cat(
                (
                    cities,
                    td["expected_prize"][..., None],
                    td["penalty"][..., 1:, None],
                ),
                -1,
            )
        )
        # batch, n_city+1, embed_dim
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out

class SymPCTSPContext(EnvContext):
    """Context embedding for the Orienteering Problem (OP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
        - remaining distance (max_length - tour_length)
    """

    def __init__(self, embed_dim):
        super(SymPCTSPContext, self).__init__(embed_dim, embed_dim + 1)
        self.project_first = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        if td["i"][(0,) * td["i"].dim()].item() < 1:   # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, -1
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], -1
                )
            self.first_context = None
            return context_embedding
        else:
            if td["i"][(0,) * td["i"].dim()].item() == 1:
                first_node_embedding = gather_by_index(embeddings, td["first_node"])
                self.first_context = self.project_first(first_node_embedding)
            
            context_embedding = torch.cat(
                [
                    self._cur_node_embedding(embeddings, td),
                    torch.clamp(
                        td["prize_required"] - td["cur_total_prize"], min=0
                    )[..., None], 
                ], 
                dim=-1,
            )
            return self.project_context(context_embedding) + self.first_context

    