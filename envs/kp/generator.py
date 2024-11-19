from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform


from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class KPGenerator(Generator):
    def __init__(
        self,
        num_items: int = 50,
        total_budget: float = 12.5,
        weight_distribution: Union[int, float, str, type, Callable] = Uniform,
        value_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        self.num_items = num_items
        self.total_budget = total_budget
        if kwargs.get("weight_sampler", None) is not None:
            self.weight_sampler = kwargs["weight_sampler"]
        else:
            self.weight_sampler = get_sampler(
                "weight", weight_distribution, 0., 1., **kwargs
            )
        if kwargs.get("value_sampler", None) is not None:
            self.value_sampler = kwargs["value_sampler"]
        else:
            self.value_sampler = get_sampler(
                "value", value_distribution, 0., 1., **kwargs
            )
    def _generate(self, batch_size) -> TensorDict:
        # Sample locations: depot and customers
        weights = self.weight_sampler.sample((*batch_size, self.num_items))
        values = self.value_sampler.sample((*batch_size, self.num_items))
        return TensorDict(
            {
                "weights": weights,
                "values": values,
                "total_budget": self.total_budget,
            },
            batch_size=batch_size,
        )
        