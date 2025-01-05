from typing import Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from .transforms import CompositeAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from .policy import ReSymPolicy
from .losses import symmetricity_loss, mixed_loss
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ReSym(REINFORCE):
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Union[str, nn.Module] = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment_train_list: list = [],
        num_augment_list: list = [],
        augment_fn_train_list: list = [],
        augment_fn_list: list = [],
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        mix_loss: bool = False,
        elite_k: int = 1,
        elite_alpha: bool = 0.,
        po_alpha: float = 0.,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy = ReSymPolicy(env_name=env.name, **policy_kwargs)

        assert baseline == "shared", "ReSym only supports shared baseline"

        super(ReSym, self).__init__(env, policy, baseline, **kwargs)

        self.num_starts = num_starts
        
        if len(num_augment_list):
            self.augment = CompositeAugmentation(
                num_augment_list=num_augment_list,
                augment_fn_list=augment_fn_list,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
            self.num_augment = self.augment.n_aug
        else:
            self.num_augment = 0
        
        if len(num_augment_train_list):
            self.augment_train = CompositeAugmentation(
                num_augment_list=num_augment_train_list,
                augment_fn_list=augment_fn_train_list,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
            self.num_augment_train = self.augment_train.n_aug
        else:
            self.num_augment_train = 0

        self.mix_loss = mix_loss
        self.elite_k = elite_k
        self.elite_alpha = elite_alpha
        self.po_alpha = po_alpha
    
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        if phase == "train" and self.current_epoch >= 100:
            if self.env.name == "pdp":
                num_loc = torch.randint(50, 251, []).item() * 2
            else:
                num_loc = torch.randint(100, 501, []).item()
            self.env.generator.num_loc = num_loc
            batch_size = 128_0000 // num_loc ** 2
            batch = batch[:batch_size]
            batch["locs"] = batch["locs"][:, :num_loc, :]
            if "op" in self.env.name:
                batch["prize"] = batch["prize"][:, :num_loc]
            if "pctsp" in self.env.name:
                max_length = torch.randint(4, 10, []).item()
                batch["penalty"] = batch["penalty"][:, :num_loc] * (max_length / 9)
                batch["deterministic_prize"] = batch["deterministic_prize"][:, :num_loc]
                batch["stochastic_prize"] = batch["stochastic_prize"][:, :num_loc]
                
            if self.current_epoch >= 200:
                self.elite_k = 20
                self.elite_alpha = 0.1
            else:
                self.elite_alpha = 0.
        else:
            self.env.generator.num_loc = batch["locs"].shape[1]

        td = self.env.reset(batch)    
        
        n_start = self.env.get_num_starts(td) if self.num_starts is None else self.num_starts

        if phase == "train" and self.current_epoch >= 100:
            n_start = num_loc + 1
        
        # During training, we use num_augment_train     
        n_aug = self.num_augment_train if phase == "train" else self.num_augment

        if n_aug > 1:
            if phase == "train":
                td = self.augment_train(td)
            else:
                td = self.augment(td)

        # Evaluate policy
        # Set multisample=True to sample solutions with same start
        decode_type = getattr(self.policy, f"{phase}_decode_type")
        
        if decode_type == "greedy":
            n_start = 0

        if "multistart" in decode_type:
            out = self.policy(
                td, env=self.env, phase=phase, num_starts=n_start, multistart=True, select_best=False,
            )
        else:
            out = self.policy(
                td, env=self.env, phase=phase, num_samples=n_start, multisample=True, select_best=False,
            )

        # Unbatchify reward to [batch_size, num_augment, num_starts].
        reward = unbatchify(out["reward"], (n_aug, n_start))

        # Training phase
        if phase == "train":
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_start))
            if n_start > 1:
                max_reward, max_idxs = reward.max(dim=-1)
                out.update({"max_reward": max_reward})
            if n_aug > 1:
                reward_ = max_reward if n_start > 1 else reward
                max_aug_reward, max_idxs = reward_.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

            self.calculate_loss(td, batch, out, reward, log_likelihood)
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
        """Calculate loss following Sym-NCO and PO

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

        if self.mix_loss:
            loss = mixed_loss(reward, log_likelihood, self.po_alpha, self.elite_k, self.elite_alpha)
        else:
            loss_ps = symmetricity_loss(reward, log_likelihood, 1, self.po_alpha, self.elite_k, self.elite_alpha)
            loss_ss = symmetricity_loss(reward, log_likelihood, -1, self.po_alpha, self.elite_k, self.elite_alpha)
            loss = loss_ps + loss_ss

        policy_out.update(
            {
                "loss": loss,
            }
        )
        return policy_out
