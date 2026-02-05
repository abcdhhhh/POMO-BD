from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger

from ..mtvrp.generator import MTVRPGenerator
from ..mtvrp.env import MTVRPEnv

log = get_pylogger(__name__)

class SymMTVRPEnv(MTVRPEnv):
    name = "symmtvrp"

    def __init__(
        self,
        generator: MTVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MTVRPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        distances = get_distance(curr_loc[..., None, :], td["locs"]) * ((curr_node[:, None] != 0) | ~td["open_route"])
        distance = gather_by_index(distances, prev_node, squeeze=False)

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times, end_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        ).permute(2, 0, 1)
        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * ~td["done"] * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance
        )

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # this holds for backhaul_classes 0, 1, and 2:
        used_capacity_linehaul = (curr_node[:, None] != 0) * ~td["done"] * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * ~td["done"] * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        is_first_route = td["is_first_route"] & (curr_node[:, None] != 0)
        first_capacity_linehaul = td["first_capacity_linehaul"] + selected_demand_linehaul * is_first_route
        first_capacity_backhaul = td["first_capacity_backhaul"] + selected_demand_backhaul * is_first_route
        first_node = curr_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        if batch_to_scalar(td["i"]) == 0:
            penalty_time = torch.minimum(end_times, td["time_windows"][:, 0:1, 1] - distance)
            latest_time = penalty_time + service_time
            first_distances = distances
            first_route_length = torch.zeros_like(td["current_route_length"])
        else:
            latest_time = td["is_first_route"] * (
                torch.clamp(td["latest_time"] + distance / td["speed"], start_times, end_times) + service_time
            )
            penalty_time = td["penalty_time"] - td["is_first_route"] * torch.maximum(
                td["latest_time"] + distance / td["speed"] - end_times, torch.zeros_like(td["latest_time"])
            )
            penalty_time = torch.nan_to_num(penalty_time, nan=torch.inf)
            first_distances = td["first_distances"]
            first_route_length = td["first_route_length"] + td["is_first_route"] * distance

        d_i1 = gather_by_index(first_distances, curr_node)[:, None]
        d_01 = first_distances[..., [0]]
        assert (d_01 / td["speed"] <= penalty_time + 1e-5).all(), "Penalty time must be greater than first distance"


        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)

        all_visited = (visited.sum(-1) == visited.size(-1))[:, None]
        can_reach_customer = (curr_time + d_i1 / td["speed"] <= penalty_time)
        exceeds_cap_linehaul = (used_capacity_linehaul + first_capacity_linehaul > td["vehicle_capacity"])
        exceeds_cap_backhaul = (used_capacity_backhaul + first_capacity_backhaul > td["vehicle_capacity"])
        wrong_order = ((selected_demand_backhaul > 0) & (first_capacity_linehaul > 0))
        meets_demand_constraint = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~wrong_order
        )
        exceeds_dist_limit = (curr_route_length + d_i1 + first_route_length > td["distance_limit"])
        done = td["done"] | (
            all_visited & (
                (curr_node[:, None] == 0)
                | (can_reach_customer
                & meets_demand_constraint
                & ~exceeds_dist_limit)
                )
        )
        reward = torch.zeros_like(
            done
        ).float()  # we use the `get_reward` method to compute the reward

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,

                "latest_time": latest_time,
                "penalty_time": penalty_time,
                "first_distances": first_distances,
                "first_route_length": first_route_length,
                "first_node": first_node,
                "is_first_route": is_first_route,
                "first_capacity_linehaul": first_capacity_linehaul,
                "first_capacity_backhaul": first_capacity_backhaul,
                "i": td["i"] + 1,

            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self,
        td: Optional[TensorDict],
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Demands: linehaul (C) and backhaul (B). Backhaul defaults to 0
        demand_linehaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), td["demand_linehaul"]],
            dim=1,
        )
        demand_backhaul = td.get(
            "demand_backhaul",
            torch.zeros_like(td["demand_linehaul"]),
        )
        demand_backhaul = torch.cat(
            [torch.zeros_like(td["demand_linehaul"][..., :1]), demand_backhaul], dim=1
        )
        # Backhaul class (MB). 1 is the default backhaul class
        backhaul_class = td.get(
            "backhaul_class",
            torch.full((*batch_size, 1), 1, dtype=torch.int32),
        )

        # Time windows (TW). Defaults to [0, inf] and service time to 0
        time_windows = td.get("time_windows", None)
        if time_windows is None:
            time_windows = torch.zeros_like(td["locs"])
            time_windows[..., 1] = float("inf")
        service_time = td.get("service_time", torch.zeros_like(demand_linehaul))

        # Open (O) route. Defaults to 0
        open_route = td.get(
            "open_route", torch.zeros_like(demand_linehaul[..., :1], dtype=torch.bool)
        )

        # Distance limit (L). Defaults to inf
        distance_limit = td.get(
            "distance_limit", torch.full_like(demand_linehaul[..., :1], float("inf"))
        )

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand_backhaul": demand_backhaul,
                "demand_linehaul": demand_linehaul,
                "backhaul_class": backhaul_class,
                "distance_limit": distance_limit,
                "service_time": service_time,
                "open_route": open_route,
                "time_windows": time_windows,
                "speed": td.get("speed", torch.ones_like(demand_linehaul[..., :1])),
                "vehicle_capacity": td.get(
                    "vehicle_capacity", torch.ones_like(demand_linehaul[..., :1])
                ),
                "capacity_original": td.get(
                    "capacity_original", torch.ones_like(demand_linehaul[..., :1])
                ),
                "current_node": torch.zeros(
                    (*batch_size,), dtype=torch.long, device=device
                ),
                "current_route_length": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for distance limits
                "current_time": torch.zeros(
                    (*batch_size, 1), dtype=torch.float32, device=device
                ),  # for time windows
                "used_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in backhaul
                "used_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),  # for capacity constraints in linehaul
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "is_first_route": torch.ones(
                    (*batch_size, 1), dtype=torch.bool, device=td.device
                ),
                "first_capacity_backhaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "first_capacity_linehaul": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "done": torch.zeros(
                    (*batch_size, 1), dtype=torch.bool, device=device
                ),
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        if batch_to_scalar(td["i"]) == 0:
            return torch.ones_like(td["visited"])
        
        curr_node = td["current_node"]  # note that this was just updated!
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw
        # we must ensure that we can return to depot in time *if* route is closed
        # i.e. start time + service time + time back to depot < late_tw
        can_reach_depot = (
            torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
        ) * ~td["open_route"] < late_tw[..., 0:1]

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Capacity constraints linehaul (C) and backhaul (B)
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        # Backhaul class 1 (classical backhaul) (B)
        # every customer is either backhaul or linehaul, all linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
            )
            > 0
        )
        meets_demand_constraint_backhaul_1 = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Backhaul class 2 (mixed pickup and delivery / mixed backhaul) (MB)
        # to serve linehaul customers we additionally need to check the remaining capacity in the vehicle
        # capacity is vehicle_capacity-used_capacity_backhauls, as all used_capacity_linehaul at this point have already been *delivered*
        cannot_serve_linehaul = (
            td["demand_linehaul"] > td["vehicle_capacity"] - td["used_capacity_backhaul"]
        )
        meets_demand_constraint_backhaul_2 = (
            ~exceeds_cap_linehaul & ~exceeds_cap_backhaul & ~cannot_serve_linehaul
        )

        # Now we merge the constraints of backhaul class 1 and 2 depending on the backhaul class
        meets_demand_constraint = (
            (td["backhaul_class"] == 1) & meets_demand_constraint_backhaul_1
        ) | ((td["backhaul_class"] == 2) & meets_demand_constraint_backhaul_2)

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, 0] = ~(curr_node == 0)

        allow = td["done"].squeeze(-1)
        can_visit[allow, :] = 0
        can_visit[allow] = can_visit[allow].scatter(-1, td["first_node"][allow, None], 1)
        
        assert (can_visit.sum(-1) > 0).all(), "No valid actions"
        return can_visit

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        actions = torch.cat([actions, actions[:, 0:1]], dim=-1)
        lengths = torch.argmax((actions == actions.roll(shifts=-1, dims=-1)).int(), dim=-1, keepdim=True)
        assert (lengths > 0).all(), "Invalid tour"
        shifts = torch.minimum(torch.argmax((actions == 0).int(), dim=-1, keepdim=True), lengths - 1)
        assert (gather_by_index(actions, shifts) == 0).all(), "Invalid tour, not visiting depot"
        n = actions.shape[-1]
        arange = torch.arange(n).to(actions.device).unsqueeze(0)
        indices = (arange + shifts + 1) % lengths 
        actions = gather_by_index(actions, indices)
        actions[arange >= lengths] = 0

        MTVRPEnv.check_solution_validity(td, actions)
    def get_num_starts(self, td):
        return super().get_num_starts(td) + 1
    
    def select_start_nodes(self, td, num_starts):
        num_loc_with_depot = td["locs"].shape[-2]
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc_with_depot
        )
        return selected