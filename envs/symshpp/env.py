from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.pylogger import get_pylogger

from ..shpp.env import SHPPEnv

log = get_pylogger(__name__)


class SymSHPPEnv(SHPPEnv):
    name = "symshpp"

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_node = td["action"]
        last_node = current_node if td["i"].all() == 0 else td["last_node"]

        num_loc = td["locs"].shape[-2]
        
        # Set not visited to 0 (i.e., we visited the node)
        available = td["available"].scatter(
            -1, current_node[..., None], 0
        )

        # If one endpoint is visited and the other is not, it should be visited then
        action_mask = available.clone()
        action_mask[(current_node == 0) & available[..., num_loc - 1]] = torch.tensor(
            [0] * (num_loc - 1) + [1],
            dtype=torch.bool,
            device=action_mask.device,
        )
        action_mask[(current_node == num_loc - 1) & available[..., 0]] = torch.tensor(
            [1] + [0] * (num_loc - 1),
            dtype=torch.bool,
            device=action_mask.device,
        )

        # We are done there are no unvisited locations
        done = torch.sum(available, dim=-1) == 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "last_node": last_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "available": available,
                "action_mask": action_mask,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        """Note: the last node is the starting node"""
        device = td.device
        locs = td["locs"]

        # We do not enforce loading from self for flexibility
        num_loc = locs.shape[-2]

        # Other variables
        current_node = torch.zeros(*batch_size, dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        action_mask = available.clone()
        i = torch.zeros(*batch_size, dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "i": i,
                "available": available,
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor):
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"
        idx_last = actions.argmin(dim=1)
        idx_last = actions.argmax(dim=1)
        assert (
            (idx_last - idx_last).abs() == 1
        ).all(), "Invalid tour"
