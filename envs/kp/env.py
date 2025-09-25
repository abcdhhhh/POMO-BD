from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from rl4co.envs.routing.tsp.env import TSPEnv
from .generator import KPGenerator

log = get_pylogger(__name__)

class KPEnv(TSPEnv):
    name = "kp"

    def __init__(
        self,
        generator: KPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = KPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]

        capacity = td["capacity"] - gather_by_index(
            td["locs"][:, :, [0]], current_node.unsqueeze(-1)
        )

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        ) & (td["locs"][:, :, 0] <= capacity)
        available[:, 0] = ~available[:, 1:].any(dim=-1)

        # We are done there are no unvisited locations
        done = torch.sum(available[:, 1:], dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_node,
                "i": td["i"] + 1,
                "capacity": capacity,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = torch.cat((torch.zeros_like(td["locs"][:, [0], :]), td["locs"]), -2)

        # We do not enforce loading from self for flexibility
        num_loc = init_locs.shape[-2]

        # Other variables
        current_node = torch.zeros((batch_size), dtype=torch.int64, device=device)
        capacity = td["capacity"]
        available = init_locs[:, :, 0] <= capacity
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": init_locs,
                "current_node": current_node,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        if self.check_solution:
            self.check_solution_validity(td, actions)

        # Gather locations in order of tour and return distance between them (i.e., -reward)
        locs_ordered = gather_by_index(td["locs"], actions)
        return locs_ordered[..., 1].sum(dim=-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: each node is visited once at most (except for depot)
        """

        # Check that tours are valid, i.e. contain 0 to n-1
        sorted_actions = actions.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"

        assert (td["capacity"] + 1e-5 >= 0).all(), "Capacity exceeded"

    @staticmethod
    def load_data(fpath, batch_size=[]):
        """Dataset loading from file
        Normalize demand by capacity to be in [0, 1]
        """
        td_load = load_npz_to_tensordict(fpath)
        td_load.set("capacity", td_load["capacity"][:, None])
        return td_load
    