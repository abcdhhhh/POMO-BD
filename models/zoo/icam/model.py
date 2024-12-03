from typing import Any, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict

from rl4co.data.transforms import StateAugmentation
from transforms import CompositeAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from .policy import ICAMPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ICAM(REINFORCE):
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
        elite_k: int = 1,
        elite_alpha: float = 0.,
        po_alpha: float = 0.,
        **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy = ICAMPolicy(env_name=env.name, **policy_kwargs)

        super(ICAM, self).__init__(env, policy, baseline, **kwargs)

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

        self.elite_k = elite_k
        self.elite_alpha = elite_alpha
        self.po_alpha = po_alpha
    
    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)    
        
        n_start = self.env.get_num_starts(td) if self.num_starts is None else self.num_starts
        
        # During training, we use num_augment_train     
        n_aug = self.num_augment_train if phase == "train" else self.num_augment

        if phase == "train" and False:
            # downsample td for curriculum training
            if self.env.name == "tsp":
                if self.current_epoch < 100:
                    num_loc = 100
                    batch_size = 256
                else:
                    num_loc = torch.randint(100, 501, []).item()
                    batch_size = 1_600_000 // num_loc ** 2
                n_start = num_loc
                td = td[:batch_size]
                td["action_mask"] = td["action_mask"][:, :num_loc]
                td["locs"] = td["locs"][:, :num_loc, :]
            elif self.env.name == "cvrp":
                if self.current_epoch < 100:
                    num_loc = 100
                    capacity = 50
                    batch_size = 128
                else:
                    num_loc = torch.randint(100, 501, []).item()
                    capacity = torch.randint(50, 101, []).item()
                    batch_size = 1_280_000 // num_loc ** 2
                n_start = num_loc
                td = td[:batch_size]
                td["action_mask"] = td["action_mask"][:, :num_loc+1]
                td["demand"] = td["demand"][:, :num_loc] * td["capacity"] / capacity
                td["capacity"].fill_(capacity)
                td["locs"] = td["locs"][:, :num_loc+1, :]
                td["visited"] = td["visited"][:, :num_loc+1]
            elif self.env.name in ["op", "symop"]:
                if self.current_epoch < 100:
                    num_loc = 100
                    batch_size = 256
                else:
                    num_loc = torch.randint(100, 501, []).item()
                    batch_size = 1_600_000 // num_loc ** 2
                n_start = num_loc
                td = td[:batch_size]
                td["action_mask"] = td["action_mask"][:, :num_loc+1]
                td["locs"] = td["locs"][:, :num_loc+1, :]
                td["prize"] = td["prize"][:, :num_loc+1]
                td["visited"] = td["visited"][:, :num_loc+1]

                if self.env.name == "op":
                    td["max_length"] = td["max_length"][:, :num_loc+1]
                    
            elif self.env.name in ["pctsp", "spctsp"]:
                if self.current_epoch < 100:
                    num_loc = 100
                    penalty_factor = 4
                    batch_size = 256
                else:
                    num_loc = torch.randint(100, 501, []).item()
                    penalty_factor = torch.randint(4, 10, []).item()
                    batch_size = 1_600_000 // num_loc ** 2
                n_start = num_loc
                td = td[:batch_size]
                td["action_mask"] = td["action_mask"][:, :num_loc+1]
                td["deterministic_prize"] = td["deterministic_prize"][:, :num_loc] * 500 / num_loc
                td["expected_prize"] = td["expected_prize"][:, :num_loc] * 500 / num_loc
                td["locs"] = td["locs"][:, :num_loc+1, :]
                td["penalty"] = td["penalty"][:, :num_loc+1] * 500 / num_loc * penalty_factor / 9
                td["cur_total_penalty"] = td["penalty"].sum(-1)
                td["real_prize"] = td["real_prize"][:, :num_loc+1] * 500 / num_loc
                td["stochastic_prize"] = td["stochastic_prize"][:, :num_loc] * 500 / num_loc
                td["visited"] = td["visited"][:, :num_loc+1]
            if self.env.name == "atsp" and not self.env.generator.tmat_class:
                dms = td["cost_matrix"] 
                for i in range(dms.size(-1)):
                    dms = torch.minimum(dms, dms[..., :, [i]] + dms[..., [i], :])
                td["cost_matrix"] = dms


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

            self.calculate_loss(td, batch, out, reward.flatten(1), log_likelihood.flatten(1))
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

        if self.po_alpha != 0:
            # following PO (https://openreview.net/forum?id=8QkpCRio53)
            preference = reward[:, :, None] > reward[:, None, :] 
            log_prob_pair = log_likelihood[:, :, None] - log_likelihood[:, None, :] 
            pf_log = torch.log(F.sigmoid(self.po_alpha * log_prob_pair)) 
            loss = -(pf_log * preference).mean()
            
        else:
            # REINFORCE baseline
            bl_val, bl_loss = (
                self.baseline.eval(td, reward, self.env) if extra is None else (extra, 0)
            )

            # Main loss function
            advantage = reward - bl_val  # advantage = reward - baseline
            reinforce_loss = -(advantage * log_likelihood).mean()

            if self.elite_alpha != 0:
                top_reward, top_idx = reward.topk(self.elite_k, dim=1)
                top_loss = -(advantage * log_likelihood).gather(1, top_idx).mean()
                loss = self.elite_alpha * top_loss + (1 - self.elite_alpha) * reinforce_loss + bl_loss
            else:
                loss = reinforce_loss + bl_loss

        policy_out.update(
            {
                "loss": loss,
            }
        )
        return policy_out
