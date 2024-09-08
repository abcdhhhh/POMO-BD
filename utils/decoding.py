from typing import Tuple
import torch

from tensordict import TensorDict

from rl4co.envs import RL4COEnvBase
from rl4co.utils.decoding import DecodingStrategy, process_logits

from rl4co.utils.ops import batchify, gather_by_index, unbatchify, unbatchify_and_gather

class DecodingStrategyForced:
    def __init__(self, strategy: DecodingStrategy):
        self.strategy = strategy
    
    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict = None,
        action: torch.Tensor = None,
        **kwargs,
    ) -> TensorDict:
        if not self.strategy.mask_logits:  # set mask_logit to None if mask_logits is False
            mask = None

        logprobs = process_logits(
            logits,
            mask,
            temperature=self.strategy.temperature,
            top_p=self.strategy.top_p,
            top_k=self.strategy.top_k,
            tanh_clipping=self.strategy.tanh_clipping,
            mask_logits=self.strategy.mask_logits,
        )
        logprobs, selected_action, td = self._step(
            logprobs, mask, td, action=action, **kwargs
        )

        # directly return for improvement methods, since the action for improvement methods is finalized in its own policy
        if self.strategy.improvement_method_mode:
            return logprobs, selected_action
        # for others
        if not self.strategy.store_all_logp:
            logprobs = gather_by_index(logprobs, selected_action, dim=1)
        td.set("action", selected_action)
        self.strategy.actions.append(selected_action)
        self.strategy.logprobs.append(logprobs)
        return td
    
    def _step(
        self,
        logprobs: torch.Tensor,
        mask: torch.Tensor,
        td: TensorDict,
        action: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        if action is not None:
            selected = action
            return torch.zeros_like(logprobs), selected, td
        else:
            return self.strategy._step(
                logprobs=logprobs,
                mask=mask, 
                td=td, 
                action=action, 
                **kwargs
            )
