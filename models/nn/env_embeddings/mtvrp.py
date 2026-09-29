import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext, VRPContext

class MTVRPInitEmbeddingRouteFinderBase(nn.Module):
    """General Init embedding class

    Args:
        num_global_feats: number of global features
        num_cust_feats: number of customer features
        embed_dim: embedding dimension
        bias: whether to use bias in the linear layers
        posinf_val: value to replace positive infinity values with
    """

    def __init__(
        self, num_global_feats, num_cust_feats, embed_dim=128, bias=False, posinf_val=0.0
    ):
        super(MTVRPInitEmbeddingRouteFinderBase, self).__init__()
        self.project_global_feats = nn.Linear(num_global_feats, embed_dim, bias=bias)
        self.project_customers_feats = nn.Linear(num_cust_feats, embed_dim, bias=bias)
        self.embed_dim = embed_dim
        self.posinf_val = posinf_val

    def _global_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _cust_feats(self, td):
        raise NotImplementedError("This method should be overridden by subclasses")

    def forward(self, td):
        global_feats = self._global_feats(td)
        cust_feats = self._cust_feats(td)

        global_feats = torch.nan_to_num(global_feats, posinf=self.posinf_val)
        cust_feats = torch.nan_to_num(cust_feats, posinf=self.posinf_val)
        global_embeddings = self.project_global_feats(global_feats)
        cust_embeddings = self.project_customers_feats(cust_feats)

        return torch.cat((global_embeddings, cust_embeddings), -2)


class MTVRPInitEmbeddingRouteFinder(MTVRPInitEmbeddingRouteFinderBase):
    """
    Customer features:
        - locs: x, y euclidean coordinates
        - demand_linehaul: demand of the nodes (delivery) (C)
        - demand_backhaul: demand of the nodes (pickup) (B)
        - time_windows: time window (TW)
        - service_time: service time of the nodes
    Global features:
        - open_route (O)
        - distance_limit (L)
        - (end) time window of depot
        - x, y euclidean coordinates of depot
    The above features are embedded in the depot node as global and get broadcasted via attention.
    This allows the network to learn the relationships between them.
    """

    def __init__(self, embed_dim=128, bias=False, posinf_val=0.0):
        super(MTVRPInitEmbeddingRouteFinder, self).__init__(
            num_global_feats=5,  # x, y, open_route, distance_limit, time_window_depot
            num_cust_feats=7,
            embed_dim=embed_dim,
            bias=bias,
            posinf_val=posinf_val,
        )

    def _global_feats(self, td):
        return torch.cat(
            [
                td["open_route"].float()[..., None],
                # TODO replace 1 with num_depots
                td["locs"][:, :1, :],
                td["distance_limit"][..., None],
                td["time_windows"][:, :1, 1:2],
            ],
            -1,
        )

    def _cust_feats(self, td):
        return torch.cat(
            (
                td["locs"][..., 1:, :],
                td["demand_linehaul"][..., 1:, None],
                td["demand_backhaul"][..., 1:, None],
                td["time_windows"][..., 1:, :],
                td["service_time"][..., 1:, None],
            ),
            -1,
        )
    
class MTVRPContextEmbeddingRouteFinder(EnvContext):
    """Context embedding MTVRP.
    - current time
    - used capacity
    - remaining distance (set to default_remain_dist if positive infinity)

    We do not need to embed the open route here since it is done encoder-side.
    """

    def __init__(self, embed_dim=128, default_remain_dist=10):
        super(MTVRPContextEmbeddingRouteFinder, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 3
        )
        self.default_remain_dist = default_remain_dist

    def _state_embedding(self, embeddings, td):
        mask = td["used_capacity_backhaul"] == 0
        used_capacity = torch.where(
            mask, td["used_capacity_linehaul"], td["used_capacity_backhaul"]
        )
        available_load = td["vehicle_capacity"] - used_capacity
        remaining_dist = torch.nan_to_num(
            td["distance_limit"] - td["current_route_length"],
            posinf=self.default_remain_dist,
        )
        context_feats = torch.cat(
            (
                available_load,
                td["current_time"],
                remaining_dist,
            ),
            -1,
        )
        return context_feats