from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

CAPACITIES = {
    50: 12.5,
    100: 25.0,
    200: 25.0,
}


class KPGenerator(Generator):
    """Data generator for the Knapsack Problem (KP).

    Args:
        num_loc: number of locations (customers) in the KP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        loc_distribution: distribution for the location coordinates

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
    """

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        init_sol_type: str = "random",
        loc_distribution: int | float | str | type | Callable = Uniform,
        capacity: float = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )
        
        # Capacity
        if (
            capacity is None
        ):  # If not provided, use the default capacity from Kool et al. 2019
            capacity = CAPACITIES.get(num_loc, None)
        if (
            capacity is None
        ):  # If not in the table keys, find the closest number of nodes as the key
            closest_num_loc = min(CAPACITIES.keys(), key=lambda x: abs(x - num_loc))
            capacity = CAPACITIES[closest_num_loc]
            log.warning(
                f"The capacity capacity for {num_loc} locations is not defined. Using the closest capacity: {capacity}\
                    with {closest_num_loc} locations."
            )
        self.capacity = capacity

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        # Sample capacities
        capacity = torch.full((*batch_size, 1), self.capacity)

        return TensorDict(
            {
                "locs": locs,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )