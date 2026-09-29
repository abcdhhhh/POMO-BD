import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext, VRPTWContext
from rl4co.models.nn.env_embeddings.init import VRPTWInitEmbedding

class SymVRPTWInitEmbedding(VRPTWInitEmbedding):
    pass

class SymVRPTWContext(EnvContext):
    def __init__(self, embed_dim):
        super(SymVRPTWContext, self).__init__(embed_dim, embed_dim * 2 + 5)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embed_dim * 2 + 5).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-2), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:   # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, -1
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], -1
                )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
                # td["current_node"],
            ).view(batch_size, *node_dim)
            
            context_embedding = torch.cat(
                [
                    context_embedding, 
                    (td["vehicle_capacity"] - td["used_capacity"]), 
                    (td["vehicle_capacity"] - td["first_capacity"]),
                    td["is_first_route"],
                    td["current_time"],
                    td["penalty_time"],
                ], 
                dim=-1,
            )
        return self.project_context(context_embedding)


    