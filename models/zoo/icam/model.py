from typing import Any, Union, Optional

import torch
import torch.nn as nn

from tensordict import TensorDict

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from .policy import ICAMPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ICAM(REINFORCE):
    """POMO Model for neural combinatorial optimization based on REINFORCE
    Based on Kwon et al. (2020) http://arxiv.org/abs/2010.16011.

    Note:
        If no policy kwargs is passed, we use the Attention Model policy with the following arguments:
        Differently to the base class:
        - `num_encoder_layers=6` (instead of 3)
        - `normalization="instance"` (instead of "batch")
        - `use_graph_context=False` (instead of True)
        The latter is due to the fact that the paper does not use the graph context in the policy, which seems to be
        helpful in overfitting to the training graph size.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that POMO only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        num_starts: Number of starts for multi-start. If None, use the number of available actions
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: Union[str, callable] = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        elite_k: int = 1,
        elite_alpha: float = 0.,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = ICAMPolicy(env_name=env.name, **policy_kwargs_with_defaults)

        assert baseline == "shared", "ICAM only supports shared baseline"

        # Initialize with the shared baseline
        super(ICAM, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        # Add `_multistart` to decode type for train, val and test in policy
        for phase in ["train", "val", "test"]:
            self.set_decode_type_multistart(phase)
        self.elite_k = elite_k
        self.elite_alpha = elite_alpha

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
        out = self.policy(
            td, self.env, phase=phase, num_starts=n_start, return_actions=True
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
                        out["best_multistart_actions"] if n_start > 1 else out["actions"]
                    )
                    out.update({"best_aug_actions": gather_by_index(actions_, max_idxs)})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
    
    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: dict,
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ):
        """Calculate loss following Poppy (https://arxiv.org/abs/2210.03475).

        Args:
            td: TensorDict containing the current state of the environment
            batch: Batch of data. This is used to get the extra loss terms, e.g., REINFORCE baseline
            policy_out: Output of the policy network
            reward: Reward tensor. If None, it is taken from `policy_out`
            log_likelihood: Log-likelihood tensor. If None, it is taken from `policy_out`
        """
        # Extra: this is used for additional loss terms, e.g., REINFORCE baseline
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = (
            log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        )

        # REINFORCE baseline
        bl_val, bl_loss = (
            self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
        )

        # Main loss function
        advantage = reward - bl_val  # advantage = reward - baseline
        reinforce_loss = -(advantage * log_likelihood).mean()

        top_reward, top_idx = reward.topk(self.elite_k, dim=1)
        top_loss = -(advantage * log_likelihood).gather(1, top_idx).mean()

        loss = self.elite_alpha * top_loss + (1 - self.elite_alpha) * reinforce_loss + bl_loss
        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            }
        )
        return policy_out
