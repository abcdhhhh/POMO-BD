from typing import Any, Union, Optional

import torch.nn as nn

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl import REINFORCE
from rl4co.models.rl.reinforce.baselines import REINFORCEBaseline
from rl4co.utils.ops import gather_by_index, unbatchify

from models.zoo.icam.model import ICAM
from .policy import UDCPolicy


class UDC(ICAM):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: UDCPolicy = None,
        policy_kwargs={},
        num_augment: int = 1,
        num_starts: int = None,
        elite_k: int = 1,
        elite_alpha: float = 0.,
        reviser: nn.Module = None,
        **kwargs,
    ):
        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = UDCPolicy(
                env_name=env.name,
                reviser=reviser,
                **policy_kwargs_with_defaults,
            )

        super().__init__(
            env=env,
            policy=policy,
            num_augment=num_augment,
            num_starts=num_starts,
            elite_k=elite_k,
            elite_alpha=elite_alpha,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        # Set multisample=True to sample solutions with same start
        out = self.policy(
            td, env=self.env, phase=phase, num_starts=n_start, multisample=True, return_actions=True
        )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            assert n_start > 1, "num_starts must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            self.calculate_loss(td, batch, out, reward, log_likelihood)
            max_reward, max_idxs = reward.max(dim=-1)
            out.update({"max_reward": max_reward})
        # Get multi-start (=POMO) rewards and best actions only during validation and test
        else:
            if n_start > 1:
                # max multi-start reward
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})

                if out.get("actions", None) is not None:
                    # Reshape batch to [batch_size, num_augment, num_starts, ...]
                    actions = unbatchify(out["actions"], (n_aug, n_start))
                    out.update(
                        {"best_multistart_actions": gather_by_index(actions, max_idxs, dim=max_idxs.dim())}
                    )
                    out["actions"] = actions

            # Get augmentation score only during inference
            if n_aug > 1:
                # If multistart is enabled, we use the best multistart rewards
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    actions_ = (
                        out["best_multistart_actions"] if n_start > 1 else unbatchify(out["actions"], n_aug)
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        if phase=="train":
            out.update({
                'divider_loss': out['loss'],
                'loss': out['loss'] + out['reviser_loss'],
            })

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}