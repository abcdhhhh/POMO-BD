from typing import Any, Union, Optional

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.ops import gather_by_index, unbatchify

from .policy import UDCPolicy


class UDC(REINFORCE):

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: UDCPolicy = None,
        baseline: Union[REINFORCEBaseline, str] = "shared",
        reviser: nn.Module = None,
        num_starts: int = None,
        policy_kwargs={},
        baseline_kwargs={},
        **kwargs,
    ):
        if policy is None:
            policy = UDCPolicy(
                env_name=env.name,
                reviser=reviser,
                **policy_kwargs,
            )

        super().__init__(env, policy, baseline, baseline_kwargs, **kwargs)
        self.num_starts = num_starts

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        num_starts = self.num_starts

        # Evaluate policy
        out = self.policy(
            td, env=self.env, phase=phase, num_starts=num_starts, multisample=True, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (num_starts))
        # Training phase
        if phase == "train":
            assert num_starts > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (num_starts))
            out = self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})

            # change loss
            out.update({"divider_loss": out["loss"], "loss": out["loss"]+out["reviser_loss"]})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if num_starts > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (num_starts))
                    out.update(
                        {"best_multistart_actions": gather_by_index(actions, max_idxs, dim=max_idxs.dim())}
                    )
                    out["actions"] = actions
        
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
   