from typing import Optional

import torch
from tqdm import tqdm

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import (
    load_npz_to_tensordict,
    load_solomon_instance,
    load_solomon_solution,
)
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length, get_distance
from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.envs.routing.cvrptw.env import CVRPTWEnv
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from .generator import SymCVRPTWGenerator
from rl4co.envs.routing.cvrptw.render import render
from envs.symcvrp.env import SymCVRPEnv

log = get_pylogger(__name__)


class SymCVRPTWEnv(SymCVRPEnv):
    """Capacitated Vehicle Routing Problem with Time Windows (CVRPTW) environment.
    Inherits from the CVRPEnv class in which customers are considered.
    Additionally considers time windows within which a service has to be started.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle.
        - the current time.
        - service durations of each location.
        - time windows of each location.

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining customer.
        - the vehicle can return to the depot to refill the customer.
        - the vehicle must start the service within the time window of each location.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPTWGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "symcvrptw"

    def __init__(
        self,
        generator: SymCVRPTWGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SymCVRPTWGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)
    

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """In addition to the constraints considered in the CVRPEnv, the time windows are considered.
        The vehicle can only visit a location if it can reach it in time, i.e. before its time window ends.
        """

        if batch_to_scalar(td["i"]) == 0:
            return torch.ones_like(td["visited"])
        
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"] + 1e-5
        
        can_reach_in_time = (
            td["current_time"] + td["distances"] <= td["time_windows"][..., 1]
        )  # I only need to start the service before the time window ends, not finish it.

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited
        mask_depot = (td["current_node"] == 0)
        mask = torch.cat((mask_depot, mask_loc), -1) | ~can_reach_in_time

        allow = td["done"].squeeze(-1)
        mask[allow, :] = 1
        mask[allow] = mask[allow].scatter(-1, td["first_node"][allow], 0)

        return ~mask

    def _step(self, td: TensorDict) -> TensorDict:
        """In addition to the calculations in the CVRPEnv, the current time is
        updated to keep track of which nodes are still reachable in time.
        The current_node is updated in the parent class' _step() function.
        """
        batch_size = td["locs"].shape[0]

        # update current_time
        distance = gather_by_index(td["distances"], td["action"]).reshape([batch_size, 1])
        duration = gather_by_index(td["durations"], td["action"]).reshape([batch_size, 1])
        start_times = gather_by_index(td["time_windows"], td["action"])[..., 0].reshape(
            [batch_size, 1]
        )
        end_times = gather_by_index(td["time_windows"], td["action"])[..., 1].reshape(
            [batch_size, 1]
        )
        # current_node is updated to the selected action
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        current_time =  (
            torch.max(td["current_time"] + distance, start_times) + duration
        ) * (current_node != 0).float() * (~td["done"]).float()

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (current_node != 0).float() * (~td["done"]).float()

        is_first_route = td["is_first_route"] & (current_node != 0)
        first_capacity = td["first_capacity"] + selected_demand * is_first_route.float()

        first_node = current_node if batch_to_scalar(td["i"]) == 0 else td["first_node"]

        # update distances
        current_loc = gather_by_index(td["locs"], current_node)
        dist = get_distance(current_loc[..., None, :], td["locs"])

        if batch_to_scalar(td["i"]) == 0:
            latest_time = end_times + duration
            penalty_time = end_times
            first_length = dist
        else:
            latest_time = is_first_route.float() * (
                torch.clamp(td["latest_time"] + distance, start_times, end_times) + duration
            )
            penalty_time = td["penalty_time"] - is_first_route.float() * torch.maximum(
                td["latest_time"] + distance - end_times, torch.zeros_like(td["latest_time"])
            )
            first_length = td["first_length"]

        d_01 = first_length[..., [0]]
        assert (d_01 <=  penalty_time + 1e-5).all(), "Penalty time must be greater than first length"
        back_length = gather_by_index(first_length, current_node)[:, None]

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node, 1)
        # SECTION: get done
        done = (visited.sum(-1) == visited.size(-1))[..., None] & (used_capacity + first_capacity <= td["vehicle_capacity"] + 1e-5) & (current_time + back_length <= penalty_time + 1e-5)
        reward = torch.zeros_like(done)

        td.update(
            {
                "distances": dist,
                "first_node": first_node,
                "current_node": current_node,
                "used_capacity": used_capacity,
                "is_first_route": is_first_route,
                "first_capacity": first_capacity,
                "current_time": current_time,
                "latest_time": latest_time,
                "penalty_time": penalty_time,
                "first_length": first_length,
                "visited": visited,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))

        assert td["action_mask"].sum(-1).all()
        
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][..., None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
                ),
                "current_time": torch.zeros(
                    *batch_size, 1, dtype=torch.float32, device=device
                ),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "vehicle_capacity": torch.full(
                    (*batch_size, 1), self.generator.vehicle_capacity, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, td["locs"].shape[-2] + 1),
                    dtype=torch.uint8,
                    device=device,
                ),
                "durations": td["durations"],
                "time_windows": td["time_windows"],
                "is_first_route": torch.ones(
                    (*batch_size, 1), dtype=torch.bool, device=td.device
                ),
                "first_capacity": torch.zeros(
                    (*batch_size, 1), device=device
                ),
                "done": torch.zeros(
                    (*batch_size, 1), dtype=torch.bool, device=device
                ),
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
            },
            batch_size=batch_size,
        )

        current_loc = gather_by_index(td_reset["locs"], td_reset["current_node"])
        dist = get_distance(current_loc[..., None, :], td_reset["locs"])
        td_reset["distances"] = dist

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """The reward is the negative tour length. Time windows
        are not considered for the calculation of the reward."""
        return super()._get_reward(td, actions)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        batch_size, graph_size = td["demand"].size()
        # distances to depot
        distances = get_distance(
            td["locs"][..., 0, :], td["locs"].transpose(0, 1)
        ).transpose(0, 1)
        # basic checks on time windows
        assert torch.all(distances >= 0.0), "Distances must be non-negative."
        assert torch.all(td["time_windows"] >= 0.0), "Time windows must be non-negative."
        assert torch.all(
            td["time_windows"][..., :, 0] + distances + td["durations"]
            <= td["time_windows"][..., 0, 1][0]  # max_time is the same for all batches
        ), "vehicle cannot perform service and get back to depot in time."
        assert torch.all(
            td["durations"] >= 0.0
        ), "Service durations must be non-negative."
        assert torch.all(
            td["time_windows"][..., 0] < td["time_windows"][..., 1]
        ), "there are unfeasible time windows"

        for k, action in tqdm(enumerate(actions)):
            n = action.size(0)
            while n > 1 and action[n - 1] == action[0]:
                n -= 1
            sorted_action = action[:n].sort()[0]
            assert (sorted_action[-graph_size:] == torch.arange(
                1, graph_size + 1, device=sorted_action.device
            )).all(), "Invalid tour"
            assert (sorted_action[:-graph_size] == 0).all(), "Invalid tour"
            l = 0
            first_capacity = 0
            while l < n and action[l] != 0:
                first_capacity += td["demand"][k, action[l] - 1]
                l += 1
                
            capacity = 0.
            t = 0.
            for i in range(l, n):
                if action[i] == 0:
                    # Reset capacity
                    capacity = 0.
                    t = 0.
                else:
                    capacity += td["demand"][k, action[i] - 1]
                    assert (
                        capacity <= td["vehicle_capacity"][k, 0] + 1e-5
                    ).all(), "Used more than capacity"
                    # Check if the vehicle arrives at the location within the time window
                    last_loc = td["locs"][k, action[i - 1]]
                    current_loc = td["locs"][k, action[i]]
                    dist = get_distance(last_loc, current_loc)
                    t = max(t + dist, td["time_windows"][k, action[i], 0])
                    assert (
                        t <= td["time_windows"][k, action[i], 1]
                    ).all(), "Arrived too late at location"
                    duration = td["durations"][k, action[i]]
                    t = t + duration
                    
            assert (
                capacity + first_capacity <= td["vehicle_capacity"][k, 0] + 1e-5
            ).all(), "Used more than capacity in first route"

            for i in range(l):
                assert action[i] != 0
                last_loc = td["locs"][k, action[i - 1 if i > 0 else n - 1]]
                current_loc = td["locs"][k, action[i]]
                dist = get_distance(last_loc, current_loc)
                t = max(t + dist, td["time_windows"][k, action[i], 0])
                assert (
                    t <= td["time_windows"][k, action[i], 1]
                ).all(), "Arrived too late at location in first route"
                duration = td["durations"][k, action[i]]
                t = t + duration

    def _make_spec(self, generator: SymCVRPTWGenerator):
        if isinstance(generator, CVRPGenerator):
            super()._make_spec(generator)
        else:
            current_time = Unbounded(shape=(1), dtype=torch.float32, device=self.device)
            current_loc = Unbounded(shape=(2), dtype=torch.float32, device=self.device)
            durations = Bounded(
                low=generator.min_time,
                high=generator.max_time,
                shape=(generator.num_loc, 1),
                dtype=torch.int64,
                device=self.device,
            )
            time_windows = Bounded(
                low=generator.min_time,
                high=generator.max_time,
                shape=(
                    generator.num_loc,
                    2,
                ),  # Each location has a 2D time window (start, end)
                dtype=torch.int64,
                device=self.device,
            )
            # Extend observation specs
            self.observation_spec = Composite(
                **self.observation_spec,
                current_time=current_time,
                current_loc=current_loc,
                durations=durations,
                time_windows=time_windows,
            )

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        render(td, actions, ax)

    # @staticmethod
    # def load_data(
    #     name: str,
    #     solomon=False,
    #     path_instances: str = None,
    #     type: str = None,
    #     compute_edge_weights: bool = False,
    # ):
    #     if solomon:
    #         assert type in [
    #             "instance",
    #             "solution",
    #         ], "type must be either 'instance' or 'solution'"
    #         if type == "instance":
    #             instance = load_solomon_instance(
    #                 name=name, path=path_instances, edge_weights=compute_edge_weights
    #             )
    #         elif type == "solution":
    #             instance = load_solomon_solution(name=name, path=path_instances)
    #         return instance
    #     return load_npz_to_tensordict(filename=name)

    # def extract_from_solomon(self, instance: dict, batch_size: int = 1):
    #     # extract parameters for the environment from the Solomon instance
    #     self.min_demand = instance["demand"][1:].min()
    #     self.max_demand = instance["demand"][1:].max()
    #     self.vehicle_capacity = instance["capacity"]
    #     self.min_loc = instance["node_coord"][1:].min()
    #     self.max_loc = instance["node_coord"][1:].max()
    #     self.min_time = instance["time_window"][:, 0].min()
    #     self.max_time = instance["time_window"][:, 1].max()
    #     # assert the time window of the depot starts at 0 and ends at max_time
    #     assert self.min_time == 0, "Time window of depot must start at 0."
    #     assert (
    #         self.max_time == instance["time_window"][0, 1]
    #     ), "Depot must have latest end time."
    #     # convert to format used in CVRPTWEnv
    #     td = TensorDict(
    #         {
    #             "depot": torch.tensor(
    #                 instance["node_coord"][0],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "locs": torch.tensor(
    #                 instance["node_coord"][1:],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1, 1),
    #             "demand": torch.tensor(
    #                 instance["demand"][1:],
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "durations": torch.tensor(
    #                 instance["service_time"],
    #                 dtype=torch.int64,
    #                 device=self.device,
    #             ).repeat(batch_size, 1),
    #             "time_windows": torch.tensor(
    #                 instance["time_window"],
    #                 dtype=torch.int64,
    #                 device=self.device,
    #             ).repeat(batch_size, 1, 1),
    #         },
    #         batch_size=1,  # we assume batch_size will always be 1 for loaded instances
    #     )
    #     return self.reset(td, batch_size=batch_size)
