from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
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
        distance = get_distance(prev_loc, curr_loc)[..., None]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * (
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
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Done when all customers are visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)
        done = visited.sum(-1) == visited.size(-1)
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
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
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
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        batch_size, n_loc = td["demand_linehaul"].size()
        locs = td["locs"]
        n_loc -= 1  # exclude depot
        sorted_pi = actions.data.sort(1)[0]

        # all customer nodes visited exactly once
        assert (
            torch.arange(1, n_loc + 1, out=sorted_pi.data.new())
            .view(1, -1)
            .expand(batch_size, n_loc)
            == sorted_pi[:, -n_loc:]
        ).all() and (sorted_pi[:, :-n_loc] == 0).all(), "Invalid tour"

        # Distance limits (L)
        assert (td["distance_limit"] >= 0).all(), "Distance limits must be non-negative."

        # Time windows (TW)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(td["service_time"] >= 0.0), "Service time must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"
        assert torch.all(
            td["time_windows"][..., :, 0] + d_j0 + td["service_time"]
            <= td["time_windows"][..., 0, 1, None]
        ), "vehicle cannot perform service and get back to depot in time."
        # check individual time windows
        curr_time = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        curr_node = torch.zeros(batch_size, dtype=torch.int64, device=td.device)
        curr_length = torch.zeros(batch_size, dtype=torch.float32, device=td.device)
        for ii in range(actions.size(1)):
            next_node = actions[:, ii]
            curr_loc = gather_by_index(td["locs"], curr_node)
            next_loc = gather_by_index(td["locs"], next_node)
            dist = get_distance(curr_loc, next_loc)

            # distance limit (L)
            curr_length = curr_length + dist * ~(
                td["open_route"].squeeze(-1) & (next_node == 0)
            )  # do not count back to depot for open route
            assert torch.all(
                curr_length <= td["distance_limit"].squeeze(-1)
            ), "Route exceeds distance limit"
            curr_length[next_node == 0] = 0.0  # reset length for depot

            curr_time = torch.max(
                curr_time + dist, gather_by_index(td["time_windows"], next_node)[..., 0]
            )
            assert torch.all(
                curr_time <= gather_by_index(td["time_windows"], next_node)[..., 1]
            ), "vehicle cannot start service before deadline"
            curr_time = curr_time + gather_by_index(td["service_time"], next_node)
            curr_node = next_node
            curr_time[curr_node == 0] = 0.0  # reset time for depot

        # Demand constraints (C) and (B) and (MB)
        # we keep track of the current picked up linehaul and backhaul
        # and the used capacity of both
        demand_l = td["demand_linehaul"].gather(dim=1, index=actions)
        demand_b = td["demand_backhaul"].gather(dim=1, index=actions)
        used_cap_l = torch.zeros_like(td["demand_linehaul"][:, 0])
        used_cap_b = torch.zeros_like(td["demand_backhaul"][:, 0])
        for ii in range(actions.size(1)):
            # reset at depot
            used_cap_l = used_cap_l * (actions[:, ii] != 0)
            used_cap_b = used_cap_b * (actions[:, ii] != 0)
            # increase counters
            used_cap_l += demand_l[:, ii]
            used_cap_b += demand_b[:, ii]

            # For backhaul_class 1 (B), we must ensure that if we are carrying backhaul, we are not picking up linehaul
            assert (
                (td["backhaul_class"] == 2)
                | (used_cap_b == 0)
                | ((td["backhaul_class"] == 1) & ~(demand_l[:, ii] > 0))
            ).all(), "Cannot pick up linehaul while carrying backhaul due to precedence constraints"

            # For backhaul_class 2 (MB), we cannot pick up linehaul if the used capacity of backhaul is already at the vehicle capacity
            # also, cannot pick up other backhauls if we are full
            assert (
                (td["backhaul_class"] == 1)
                | (used_cap_b == 0)
                | (
                    (td["backhaul_class"] == 2)
                    & (used_cap_b + demand_l[:, ii] <= td["vehicle_capacity"])
                )
            ).all(), "Cannot deliver linehaul, not enough load"

            # Assertions: total used linehaul and backhaul capacity should not exceed vehicle capacity
            assert (
                used_cap_l <= td["vehicle_capacity"]
            ).all(), "Used more linehaul than capacity: {} / {}".format(
                used_cap_l, td["vehicle_capacity"]
            )
            assert (
                used_cap_b <= td["vehicle_capacity"]
            ).all(), "Used more backhaul than capacity: {} / {}".format(
                used_cap_b, td["vehicle_capacity"]
            )

    def get_num_starts(self, td):
        return super().get_num_starts() + 1
    
    def select_start_nodes(self, td, num_starts):
        num_loc_with_depot = td["locs"].shape[-2]
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc_with_depot
        )
        return selected