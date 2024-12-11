from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index, get_tour_length

from rl4co.utils.pylogger import get_pylogger

from ..sympdp.env import SymPDPEnv

log = get_pylogger(__name__)

class SymPDPLIFOEnv(SymPDPEnv):
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
        level = torch.zeros_like(available, dtype=int)

        # Other variables
        current_node = torch.zeros(*batch_size, 1, dtype=torch.int64, device=device)
        i = torch.zeros(*batch_size, dtype=torch.int64, device=device)

        td_reset = TensorDict(
            {
                "locs": locs,
                "current_node": current_node,
                "available": available,
                "to_deliver": to_deliver,
                "level": level,
                "i": i,
            },
            batch_size=batch_size,
        )
    
        td_reset.set("action_mask", self.get_action_mask(td_reset))

        return td_reset
    
    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        action_mask = td["available"]
        return action_mask