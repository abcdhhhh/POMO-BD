from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# from rl4co.envs.routing.pdp.generator import PDPGenerator

class SymPDPGenerator(Generator):
    """Data generator for the Pickup and Delivery Problem (PDP).
    Args:
        num_loc: number of locations (customers) in the PDP, without the depot. (e.g. 10 means 10 locs + 1 depot)
            - 1 depot
            - `num_loc` / 2 pickup locations
            - `num_loc` / 2 delivery locations
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        init_sol_type: the method type used for generating initial solutions (random or greedy)
        loc_distribution: distribution for the location coordinates
        depot_distribution: distribution for the depot location. If None, sample the depot from the locations

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each customer
            depot [batch_size, 2]: location of the depot
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        init_sol_type: str = "random",
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = None,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.init_sol_type = init_sol_type

        # Number of locations must be even
        if num_loc % 2 != 0:
            log.warning(
                "Number of locations must be even. Adding 1 to the number of locations."
            )
            self.num_loc += 1

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = get_sampler(
                "depot", depot_distribution, min_loc, max_loc, **kwargs
            ) if depot_distribution is not None else None

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        if self.depot_sampler is not None:
            depot = self.depot_sampler.sample((*batch_size, 2))
            locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        else:
            # if depot_sampler is None, sample the depot from the locations
            locs = self.loc_sampler.sample((*batch_size, self.num_loc + 1, 2))
            depot = locs[..., 0, :]
            locs = locs[..., 1:, :]

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
            },
            batch_size=batch_size,
        )
