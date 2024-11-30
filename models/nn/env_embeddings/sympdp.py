import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext

class SymPDPInitEmbedding(nn.Module):
    """Initial embedding for the Pickup and Delivery Problem (PDP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the nodes (depot, pickups and deliveries separately)
           Note that pickups and deliveries are interleaved in the input.
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(SymPDPInitEmbedding, self).__init__()
        node_dim = 2  # x, y
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        self.init_embed_pick = nn.Linear(node_dim * 2, embed_dim, linear_bias)
        self.init_embed_delivery = nn.Linear(node_dim, embed_dim, linear_bias)

    def forward(self, td):
        depot, locs = td["locs"][..., 0:1, :], td["locs"][..., 1:, :]
        num_locs = locs.size(-2)
        pick_feats = torch.cat(
            [locs[:, : num_locs // 2, :], locs[:, num_locs // 2 :, :]], -1
        )  # [batch_size, graph_size//2, 4]
        delivery_feats = locs[:, num_locs // 2 :, :]  # [batch_size, graph_size//2, 2]
        depot_embeddings = self.init_embed_depot(depot)
        pick_embeddings = self.init_embed_pick(pick_feats)
        delivery_embeddings = self.init_embed_delivery(delivery_feats)
        # concatenate on graph size dimension
        return torch.cat([depot_embeddings, pick_embeddings, delivery_embeddings], -2)

class SymPDPContext(EnvContext):
    """Context embedding for the Pickup and Delivery Problem (PDP).
    Project the following to the embedding space:
        - last node embedding
        - current node embedding
    """

    def __init__(self, embed_dim):
        super(SymPDPContext, self).__init__(embed_dim, embed_dim)
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
            if self.first_context is None:
                first_node_embedding = gather_by_index(embeddings, td["first_node"])
                self.first_context = self.project_first(first_node_embedding)
            
            context_embedding = self._cur_node_embedding(embeddings, td)
            return self.project_context(context_embedding) + self.first_context

    