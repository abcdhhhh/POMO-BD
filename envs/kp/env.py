from typing import Optional
import torch
import torch.nn.functional as F

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.envs.common.utils import batch_to_scalar

from rl4co.data.dataset import FastTdDataset
from rl4co.envs.common.base import RL4COEnvBase

from .generator import KPGenerator

class KPEnv(RL4COEnvBase):
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

    def _step(self, td: TensorDict) -> TensorDict:
        current_node = td["action"]
        current_weight = gather_by_index(td["weights"], current_node)

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        device = td.device

        td_reset = TensorDict(
            
        )