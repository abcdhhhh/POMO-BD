from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger
from rl4co.envs.common.utils import batch_to_scalar
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.envs.routing.cvrp.render import render

log = get_pylogger(__name__)


class SymCVRPEnv(CVRPEnv):
    """Capacitated Vehicle Routing Problem (CVRP) environment.
    At each step, the agent chooses a customer to visit depending on the current location and the remaining capacity.
    When the agent visits a customer, the remaining capacity is updated. If the remaining capacity is not enough to
    visit any customer, the agent must go back to the depot. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path: maximizing the reward is equivalent to minimizing the path length.

    Observations:
        - location of the depot.
        - locations and demand of each customer.
        - current location of the vehicle.
        - the remaining customer of the vehicle,

    Constraints:
        - the tour starts and ends at the depot.
        - each customer must be visited exactly once.
        - the vehicle cannot visit customers exceed the remaining capacity.
        - the vehicle can return to the depot to refill the capacity.

    Finish Condition:
        - the vehicle has visited all customers and returned to the depot.

    Reward:
        - (minus) the negative length of the path.

    Args:
        generator: CVRPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "symcvrp"

    def __init__(
        self,
        generator: CVRPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = CVRPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][:, None]  # Add dimension for step
        n_loc = td["demand"].size(-1)  # Excludes depot

        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = gather_by_index(
            td["demand"], torch.clamp(current_node - 1, 0, n_loc - 1), squeeze=False
        )

        # Increase capacity if depot is not visited, otherwise set to 0
        used_capacity = (td["used_capacity"] + selected_demand) * (current_node != 0).float() * (~td["done"]).float()

        is_first_route = td["is_first_route"] & (current_node != 0)
        first_capacity = td["first_capacity"] + selected_demand * is_first_route.float()

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        visited = td["visited"].scatter(-1, current_node, 1)

        # SECTION: get done
        done = (visited.sum(-1) == visited.size(-1))[..., None] & (used_capacity + first_capacity <= td["vehicle_capacity"] + 1e-5)
        reward = torch.zeros_like(done)

        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "used_capacity": used_capacity,
                "is_first_route": is_first_route,
                "first_capacity": first_capacity,
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
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": torch.cat((td["depot"][:, None, :], td["locs"]), -2),
                "demand": td["demand"],
                "current_node": torch.zeros(
                    *batch_size, 1, dtype=torch.long, device=device
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
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        if batch_to_scalar(td["i"]) == 0:
            return torch.ones_like(td["visited"], dtype=torch.bool)
        
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = td["demand"] + td["used_capacity"] > td["vehicle_capacity"] + 1e-5

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited
        mask_depot = (td["current_node"] == 0)
        mask = torch.cat((mask_depot, mask_loc), -1)

        allow = td["done"].squeeze(-1)
        mask[allow, :] = 1
        mask[allow] = mask[allow].scatter(-1, td["first_node"][allow], 0)
        
        return ~mask

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        # Gather locations in order of tour (add depot since we start and end there)
        locs_ordered = gather_by_index(td["locs"], actions)
        return -get_tour_length(locs_ordered)

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
        
        CVRPEnv.check_solution_validity(td, actions)

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set("demand", td_load["demand"] / td_load["capacity"][:, None])
        return td_load

    def _make_spec(self, generator: CVRPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc + 1, 2),
                dtype=torch.float32,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            demand=Bounded(
                low=-generator.capacity,
                high=generator.max_demand,
                shape=(generator.num_loc, 1),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
                dtype=torch.bool,
            ),
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            is_first_route=Unbounded(
                shape=(1),
                dtype=torch.bool,
            ),
            first_capacity=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            used_capacity=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            vehicle_capacity=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            visited=Unbounded(
                shape=(generator.num_loc + 1,),
                dtype=torch.uint8,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def replace_selected_actions(
        self,
        cur_actions: torch.Tensor,
        new_actions: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace selected current actions with updated actions based on `selection_mask`.

        Args:
            cur_actions [batch_size, num_loc]
            new_actions [batch_size, num_loc]
            selection_mask [batch_size,]
        """
        diff_length = cur_actions.size(-1) - new_actions.size(-1)
        if diff_length > 0:
            new_actions = torch.nn.functional.pad(
                new_actions, (0, diff_length, 0, 0), mode="constant", value=0
            )
        elif diff_length < 0:
            cur_actions = torch.nn.functional.pad(
                cur_actions, (0, -diff_length, 0, 0), mode="constant", value=0
            )
        cur_actions[selection_mask] = new_actions[selection_mask]
        return cur_actions
    
    def select_start_nodes(self, td, num_starts):
        num_loc_with_depot = td["locs"].shape[-2]
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc_with_depot
        )
        return selected

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax, skip_depot=False)
