from typing import Optional
import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.envs.common.utils import batch_to_scalar
from einops import rearrange

from rl4co.data.dataset import FastTdDataset
from rl4co.envs.common.base import RL4COEnvBase

from .generator import SymOPGenerator
from .render import render

# from rl4co.envs.routing.op.env import OPEnv

class SymOPEnv(RL4COEnvBase):
    """Orienteering Problem (OP) environment.
    At each step, the agent chooses a location to visit in order to maximize the collected prize.
    The total length of the path must not exceed a given threshold.

    Observations:
        - location of the depot
        - locations and prize of each customer
        - current location of the vehicle
        - current tour length
        - current total prize
        - the remaining length of the path

    Constraints:
        - the tour starts and ends at the same customer
        - not all customers need to be visited, but the depot needs
        - the vehicle cannot visit customers exceed the remaining length of the path

    Finish Condition:
        - the vehicle back to the starting point

    Reward:
        - the sum of the prizes of visited nodes

    Args:
        generator: OPGenerator instance as the data generator
        generator_params: parameters for the generator
    """
    name = "symop"

    def __init__(
        self,
        generator: SymOPGenerator = None,
        generator_params: dict = {},
        prize_type: str = "dist",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SymOPGenerator(**generator_params)
        self.generator = generator
        self.prize_type = prize_type
        assert self.prize_type in [
            "dist",
            "unif",
            "const",
        ], f"Invalid prize_type: {self.prize_type}"
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        current_loc = gather_by_index(td["locs"], current_node)
        if batch_to_scalar(td["i"]) < 1:
            first_node = current_node
            # max_length is max length allowed when arriving at node, so subtract distance to visit depot and get back
            # Additionally, substract epsilon margin for numeric stability
            first_depot_length = (td["locs"] - td["locs"][..., 0:1, :]).norm(p=2, dim=-1) + (td["locs"][..., 0:1, :] - current_loc[..., None, :]).norm(p=2, dim=-1)
            first_length = (td["locs"] - current_loc[..., None, :]).norm(p=2, dim=-1) 
            td.update(
                {
                    "first_node": first_node,
                    "first_depot_length": first_depot_length,
                    "first_length": first_length,
                }
            )
        else:
            # Update tour length
            previous_loc = gather_by_index(td["locs"], td["current_node"])  
            tour_length = td["tour_length"] + (current_loc - previous_loc).norm(p=2, dim=-1)
            td.update(
                {
                    "tour_length": tour_length,
                }
            )

        # Set current node as visited
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # Done if gone back (except if it's the first step)
        done = (current_node == td["first_node"]) & (td["i"] > 0)

        reward = gather_by_index(
            td["prize"], current_node, dim=-1
        )
        reward[done] = 0

        # Update prize with collected prize
        current_total_prize = td["current_total_prize"] + reward

        td.update(
            {
                "current_node": current_node,
                "visited": visited,
                "current_total_prize": current_total_prize,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td
    
    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        # Add depot to locs
        locs_with_depot = torch.cat((td["depot"][:, None, :], td["locs"]), -2)

        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": locs_with_depot,
                "prize": F.pad(
                    td["prize"], (1, 0), mode="constant", value=0
                ),  # add 0 for depot
                "tour_length": torch.zeros(*batch_size, device=device),
                "max_length": td["max_length"],
                "current_node": torch.zeros(
                    *batch_size, dtype=torch.long, device=device
                ),
                "visited": torch.zeros(
                    (*batch_size, locs_with_depot.shape[-2]),
                    dtype=torch.bool,
                    device=device,
                ),
                "current_total_prize": torch.zeros(
                    *batch_size, dtype=torch.float, device=device
                ),
                "i": torch.zeros(
                    (*batch_size,), dtype=torch.int64, device=device
                ),  # counter
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        """Get action mask with 1 = feasible action, 0 = infeasible action.
        Cannot visit if already visited, if depot has been visited, or if the length exceeds the maximum length.
        """
        if batch_to_scalar(td["i"]) == 0:
            depot_length = (td["locs"] - td["locs"][..., 0:1, :]).norm(p=2, dim=-1)
            return depot_length * 2 + 1e-6 <= td["max_length"][..., None]
        
        current_loc = gather_by_index(td["locs"], td["current_node"])[..., None, :]
        step_length = (current_loc - td["locs"]).norm(p=2, dim=-1)

        back_length = td["first_depot_length"]
        allow = td["visited"][..., 0]
        back_length[allow] = td["first_length"][allow]

        exceeds_length = (
            td["tour_length"][..., None] + step_length + back_length + 1e-6
            > td["max_length"][..., None]
        )
        mask = td["visited"] | td["done"][..., None] | exceeds_length
        mask[allow, td["first_node"][allow]] = 0
        mask[~allow, 0] = 0
        
        action_mask = ~mask  # 1 = feasible action, 0 = infeasible action
        return action_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Reward is the sum of the prizes of visited nodes"""
        # Prize is the sum of the prizes of the visited nodes.
        visited = torch.zeros_like(td["visited"]).scatter(-1, actions, True)
        # assert (visited == td["visited"]).all()

        current_total_prize = (td["prize"] * visited).sum(dim=-1)
        # assert torch.allclose(current_total_prize, td["current_total_prize"])
        
        return current_total_prize

    @staticmethod
    def check_solution_validity(
        td: TensorDict, actions: torch.Tensor
    ) -> None:
        """Check that solution is valid: nodes are not visited twice except depot and capacity is not exceeded.
        If `add_distance_to_depot` if True, then the distance to the depot is added to max length since by default, the max length is
        modified in the reset function to account for the distance to the depot.
        """

        # Check that tours are valid, i.e. contain 0 to n -1
        first_node = actions[..., 0]
        sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for first node)
        assert (
            (sorted_actions[:, 1:] == first_node[..., None])
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"
        assert (sorted_actions[:, 0] == 0).all(), "Depot not visited"

        # Gather locations in order of tour and get the length of tours
        locs_ordered = gather_by_index(td["locs"], actions)
        length = get_tour_length(locs_ordered)

        max_length = td["max_length"]
        assert (
            length <= max_length + 1e-5
        ).all(), "Max length exceeded by {}".format(
            (length - max_length).max()
        )

    def _make_spec(self, generator: SymOPGenerator):
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
            prize=Unbounded(
                shape=(generator.num_loc,),
                dtype=torch.float32,
            ),
            tour_length=Unbounded(
                shape=(generator.num_loc,),
                dtype=torch.float32,
            ),
            first_node=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            first_length=Unbounded(
                shape=(generator.num_loc + 1,),
                dtype=torch.float32,
            ),
            first_depot_length=Unbounded(
                shape=(generator.num_loc + 1,),
                dtype=torch.float32,
            ),
            max_length=Unbounded(
                shape=(1,),
                dtype=torch.float32,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc + 1, 1),
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

    def select_start_nodes(self, td, num_starts):
        num_loc_with_depot = td["locs"].shape[-2]
        selected = (
            torch.arange(num_starts, device=td.device).repeat_interleave(td.shape[0])
            % num_loc_with_depot
        )
        if (td["action_mask"].float().sum(-1) < num_loc_with_depot).any():
            # for the orienteering problem, we may have some nodes that are not available
            # so we need to resample from the distribution of available nodes
            selected = torch.multinomial(
                td["action_mask"].float(), num_starts, replacement=True
            )
            selected = rearrange(selected, "b n -> (n b)")
        return selected

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
