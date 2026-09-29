from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.routing.cvrp.generator import CVRPGenerator
from rl4co.envs.routing.cvrptw.generator import CVRPTWGenerator
from rl4co.utils.ops import get_distance


class SymCVRPTWGenerator(CVRPGenerator):
    """
    Following MVMoE
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        depot_distribution: int | float | str | type | Callable = Uniform,
        min_demand: int = 1,
        max_demand: int = 10,
        demand_distribution: int | float | str | type | Callable = Uniform,
        vehicle_capacity: float = 1.0,
        capacity: float = None,
        max_time: float = 3.0,
        # scale: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            depot_distribution=depot_distribution,
            min_demand=min_demand,
            max_demand=max_demand,
            demand_distribution=demand_distribution,
            vehicle_capacity=vehicle_capacity,
            capacity=capacity,
            **kwargs,
        )
        self.max_loc = max_loc
        self.min_time = 0.0
        self.max_time = max_time
        # self.scale = scale

    def _generate(self, batch_size) -> TensorDict:
        td = super()._generate(batch_size)

        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size

        ## define service durations
        durations = torch.full(size=(*batch_size, self.num_loc + 1), fill_value=0.2, dtype=torch.float32)

        ## define time windows
        # 1. get distances from depot
        dist = get_distance(td["depot"], td["locs"].transpose(0, 1)).transpose(0, 1)
        dist = torch.cat((torch.zeros(*batch_size, 1), dist), dim=1)

        # 2. define upper bound for time windows to make sure the vehicle can get back to the depot in time
        upper_bound = self.max_time - dist - durations

        # 3. time centers
        time_centers = torch.rand(*batch_size, self.num_loc + 1) * (upper_bound - dist) + dist

        # 4. time windows
        time_half_width = torch.rand(*batch_size, self.num_loc + 1) * (self.max_time / 3 - durations / 2) + durations / 2

        # 5. set min and max times
        min_times = torch.maximum(time_centers - time_half_width, dist)
        max_times = torch.minimum(time_centers + time_half_width, upper_bound)

        if not torch.all(min_times < max_times):
            return self._generate(batch_size) # retry if infeasible
        
        # 6. reset times for depot
        min_times[..., :, 0] = 0.0
        max_times[..., :, 0] = self.max_time + 1e-5

        # 7. stack to tensor time_windows
        time_windows = torch.stack((min_times, max_times), dim=-1)

        assert torch.all(
            min_times < max_times
        ), "Please make sure the relation between max_loc and max_time allows for feasible solutions."

        # Reset duration at depot to 0
        durations[:, 0] = 0.0
        td.update(
            {
                "durations": durations,
                "time_windows": time_windows,
            }
        )
        return td
