import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext, VRPContext
from .mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPContextEmbeddingRouteFinder


class SymMTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingRouteFinder):
    pass

# class SymMTVRPContextEmbeddingRouteFinder(MTVRPContextEmbeddingRouteFinder):
#     pass
    
class SymMTVRPContextEmbeddingRouteFinder(EnvContext):
    """Context embedding SymMTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(SymMTVRPContextEmbeddingRouteFinder, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim * 2 + 7
        )
        self.default_remain_dist = default_remain_dist
        self.W_placeholder = nn.Parameter(
            torch.Tensor(self.embed_dim * 2 + 7).uniform_(-1, 1)
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["current_node"].dim() == 1 else (td["current_node"].size(-1), -1)
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
            mask = td["used_capacity_backhaul"] == 0
            used_capacity = torch.where(
                mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
            )
            first_capacity = torch.where(
                mask, td["first_capacity_linehaul"], td["first_capacity_backhaul"]
            )
            available_load = td["vehicle_capacity"] - used_capacity
            first_load = td["vehicle_capacity"] - first_capacity
            remaining_dist = torch.nan_to_num(
                td["distance_limit"] - td["current_route_length"],
                posinf=self.default_remain_dist,
            )
            first_dist = torch.nan_to_num(
                td["distance_limit"] - td["first_route_length"],
                posinf=self.default_remain_dist,
            )
            penalty_time = torch.nan_to_num(
                td["penalty_time"],
                posinf=self.default_remain_dist,
            )
            context_feats = torch.cat(
                (
                    td["is_first_route"],
                    available_load,
                    first_load,
                    td["current_time"],
                    penalty_time,
                    remaining_dist,
                    first_dist,
                ),
                -1,
            )
            context_embedding = torch.cat(
                [
                    context_embedding, 
                    context_feats,
                ], 
                dim=-1,
            )
        return self.project_context(context_embedding)