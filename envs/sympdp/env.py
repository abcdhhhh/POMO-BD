from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.envs.common.utils import batch_to_scalar

from .generator import SymPDPGenerator
from .render import render

# from rl4co.envs.routing.pdp.env import PDPEnv


class SymPDPEnv(RL4COEnvBase):
    """Pickup and Delivery Problem (PDP) environment.
    The environment is made of num_loc + 1 locations (cities):
        - 1 depot
        - `num_loc` / 2 pickup locations
        - `num_loc` / 2 delivery locations
    The goal is to visit all the pickup and delivery locations in the shortest path possible starting from the depot
    The conditions is that the agent must visit a pickup location before visiting its corresponding delivery location

    Observations:
        - locations of the depot, pickup, and delivery locations
        - current location of the vehicle
        - the remaining locations to deliver
        - the remaining locations to deliver before depot
        - the visited locations
        - the current step

    Constraints: after rotation
        - the tour starts and ends at the depot
        - each pickup location must be visited before its corresponding delivery location
        - the vehicle cannot visit the same location twice

    Finish Condition:
        - the vehicle has visited all locations

    Reward:
        - (minus) the negative length of the path

    Args:
        generator: PDPGenerator instance as the data generator
        generator_params: parameters for the generator
        force_start_at_depot: whether to force the agent to start at the depot
            If False (default), the agent won't consider the depot, which is added in the `get_reward` method
            If True, the only valid action at the first step is to visit the depot (=0)
    """

    name = "sympdp"

    def __init__(
        self,
        generator: SymPDPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SymPDPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"][..., None]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        num_loc = td["locs"].shape[-2] - 1  # except depot

        # Pickup and delivery node pair of selected node
        is_delivery = current_node > num_loc // 2
        new_to_deliver = (current_node + num_loc // 2 - 1) % num_loc + 1
        

        # Set available to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node, 0
        )
        
        to_deliver = td["to_deliver"].scatter(
            -1, new_to_deliver, ~is_delivery
        )
        to_deliver[current_node[..., 0] == 0] = torch.tensor(
            [0] + [1] * (num_loc // 2) + [0] * (num_loc // 2),
            dtype=torch.bool,
            device=to_deliver.device,
        )

        expedit = td["expedit"].scatter(
            -1, current_node, 0
        ).scatter(
            -1, new_to_deliver, ~is_delivery
        )

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)
        
        # Update step
        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "available": available,
                "to_deliver": to_deliver,
                "expedit": expedit,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))

        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        locs = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # available = not visited
        available = torch.ones(
            (*batch_size, self.generator.num_loc + 1), dtype=torch.bool, device=device
        )
        to_deliver = torch.ones_like(available)
        expedit = torch.zeros_like(available)

        # Other variables
        current_node = torch.zeros(*batch_size, 1, dtype=torch.int64, device=device)
        i = torch.zeros(*batch_size, dtype=torch.int64, device=device)

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "available": available,
                "to_deliver": to_deliver,
                "expedit": expedit,
                "i": i,
            },
            batch_size=batch_size,
        )
    
        td_reset.set("action_mask", self.get_action_mask(td_reset))

        return td_reset
    
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        action_mask = td["available"] & td["to_deliver"]
        action_mask[td["expedit"].any(-1), 0] = 0
        return action_mask
    
    @staticmethod
    def _get_reward(td, actions) -> TensorDict:
        locs_ordered = gather_by_index(td["locs"], actions)  # order locations
        return -get_tour_length(locs_ordered)

    def check_solution_validity(self, td, actions):
        assert actions.size(1) == self.generator.num_loc + 1, "Not visiting all nodes once"

        assert (
            (torch.arange(actions.size(1), out=actions.data.new()))
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Not visiting all nodes"
        
        
        visited_time = torch.argsort(
            actions, 1
        )  # index of pickup less than index of delivery

        visited_time = (visited_time - visited_time[:, 0:1] + actions.size(1)) % actions.size(1) 

        assert (
            visited_time[:, 1 : actions.size(1) // 2 + 1]
            < visited_time[:, actions.size(1) // 2 + 1 :]
        ).all(), "Deliverying without pick-up"
    
    def _make_spec(self, generator: SymPDPGenerator):
        """Make the observation and action specs from the parameters."""
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
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            available=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            to_deliver=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            expedit=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
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

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
