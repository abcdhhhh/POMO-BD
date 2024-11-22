from typing import Optional
import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.envs.common.utils import batch_to_scalar

from rl4co.data.dataset import FastTdDataset
from rl4co.envs.common.base import RL4COEnvBase

from .generator import SymPCTSPGenerator
from .render import render
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# from rl4co.envs.routing.pctsp.env import PCTSPEnv


class SymPCTSPEnv(RL4COEnvBase):
    """Prize-collecting TSP (PCTSP) environment.
    The goal is to collect as much prize as possible while minimizing the total travel cost.
    The environment is stochastic, the prize is only revealed when the node is visited.

    Observations:
        - locations of the nodes
        - prize and penalty of each node
        - current location of the vehicle
        - current total prize
        - current total penalty
        - visited nodes
        - prize required to visit a node
        - the current step

    Constraints:
        - the tour starts and ends at the depot
        - the vehicle cannot visit nodes exceed the remaining prize

    Finish Condition:
        - the vehicle back to the depot

    Reward:
        - the sum of the saved penalties

    Args:
        generator: OPGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "spctsp"
    _stochastic = False

    def __init__(
        self,
        generator: SymPCTSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SymPCTSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]

        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        # Update visited
        visited = td["visited"].scatter(-1, current_node[..., None], 1)

        # Done and reward
        done = (current_node == td["first_node"]) & (td["i"] > 0)

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        # Get current coordinates, prize, and penalty
        prize = gather_by_index(
            td["real_prize"], current_node
        )
        prize[done] = 0
        cur_total_prize = td["cur_total_prize"] + prize

        penalty = gather_by_index(
            td["penalty"], current_node
        )
        penalty[done] = 0
        cur_total_penalty = td["cur_total_penalty"] + penalty

        # Update state
        td.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "cur_total_prize": cur_total_prize,
                "cur_total_penalty": cur_total_penalty,
                "visited": visited,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device

        locs = torch.cat([td["depot"][..., None, :], td["locs"]], dim=-2)
        expected_prize = td["deterministic_prize"]
        real_prize = (
            td["stochastic_prize"] if self.stochastic else td["deterministic_prize"]
        )
        penalty = td["penalty"]

        # Concatenate depots
        real_prize_with_depot = torch.cat(
            [torch.zeros_like(real_prize[..., :1]), real_prize], dim=-1
        )
        penalty_with_depot = F.pad(penalty, (1, 0), mode="constant", value=0)

        # Initialize the current node and  prize / penalty
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        cur_total_prize = torch.zeros(*batch_size, device=device)
        cur_total_penalty = penalty.sum(-1)  # Sum penalties (all when nothing is visited)

        # Init the action mask (all nodes are available)
        visited = torch.zeros(
            (*batch_size, self.generator.num_loc + 1), dtype=torch.bool, device=device
        )
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        prize_required = torch.full(
            (*batch_size,), self.generator.prize_required, device=device
        )

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "first_node": current_node,
                "expected_prize": expected_prize,
                "real_prize": real_prize_with_depot,
                "penalty": penalty_with_depot,
                "cur_total_prize": cur_total_prize,
                "cur_total_penalty": cur_total_penalty,
                "visited": visited,
                "prize_required": prize_required,
                "i": i,
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        if batch_to_scalar(td["i"]) == 0:
            return torch.ones_like(td["visited"])
        mask = td["visited"] | td["done"][..., None]
        
        """Cannot return if not yet collected 1 total prize and there are unvisited nodes"""
        not_collected = (td["cur_total_prize"] < 1.0) & (
            td["visited"][..., 1:].int().sum(-1) < td["visited"][..., 1:].size(-1)
        )
        allow = ~not_collected & ((td["first_node"] == 0) | td["visited"][..., 0])
        mask[allow, td["first_node"][allow]] = 0
        return ~mask  # Invert mask, since 1 means feasible action

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Reward is `saved penalties - (total length + penalty)`"""
        # Gather locations in order of tour
        locs_ordered = gather_by_index(td["locs"], actions)  # order locations
            
        length = get_tour_length(locs_ordered)

        # Reward is saved penalties - (total length + penalty)
        unvisited = torch.ones_like(td["visited"]).scatter(-1, actions, False)
        penalty = (td["penalty"] * unvisited).sum(dim=-1)
        
        return -(length + penalty)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that the solution is valid, i.e. contains all nodes once at most, and either prize constraint is met or all nodes are visited"""

        # Check that tours are valid, i.e. contain 0 to n -1
        last_node = actions[..., 0]
        sorted_actions = actions.data.sort(1)[0]

        # Make sure each node visited once at most (except for last node)
        assert (
            (sorted_actions[..., 1:] == last_node[..., None])
            | (sorted_actions[..., 1:] > sorted_actions[..., :-1])
        ).all(), "Duplicates"

        prize = td["real_prize"][..., 1:]  # Remove depot
        prize_with_depot = torch.cat((torch.zeros_like(prize[:, :1]), prize), 1)

        visited = torch.zeros_like(td["visited"]).scatter(-1, actions, True)
        # assert (visited == td["visited"]).all()

        cur_total_prize = (td["real_prize"] * visited).sum(dim=-1)
        # assert torch.allclose(cur_total_prize, td["cur_total_prize"])

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (cur_total_prize >= 1 - 1e-5)
            | (
                sorted_actions.size(-1) - (sorted_actions == last_node[..., None]).int().sum(-1)
                == (td["locs"].size(-2) - 1)
            )  # no depot
        ).all(), "Total prize does not satisfy min total prize"

    @property
    def stochastic(self):
        return self._stochastic

    @stochastic.setter
    def stochastic(self, state: bool):
        if state is True:
            log.warning(
                "Stochastic mode should not be used for PCTSP. Use SPCTSP instead."
            )

    def _make_spec(self, generator):
        """Make the locs and action specs from the parameters."""
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
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
            expected_prize=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.float32,
            ),
            real_prize=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.float32,
            ),
            penalty=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.float32,
            ),
            cur_total_prize=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            cur_total_penalty=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            visited=Unbounded(
                shape=(generator.num_loc + 1),
                dtype=torch.bool,
            ),
            prize_required=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def render(td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)
