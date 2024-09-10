from typing import Any

import torch
import torch.nn as nn

from rl4co.envs import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.utils.ops import gather_by_index, unbatchify, batchify

from tensordict import TensorDict
from einops import rearrange
from .action_utils import get_action_utils
from .policy import get_policy

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class UDC(REINFORCE): 
    def __init__(
        self,
        env: RL4COEnvBase,    # self & divider
        env_r: RL4COEnvBase,
        policy_d_name: str,    # divider
        policy_r_name: str,    # reviser
        policy_d_kwargs = {},
        policy_r_kwargs = {},
        alpha_train: int = 80,
        beta_train: int = 50,
        alpha_val: int = 50,
        beta_val: int = 2,
        iters: int = 2,
        sub_norm: bool = True,
        w: float = 1., 
        baseline: str = "shared",  
        num_loc_min: int = None,
        num_loc_max: int = None,  
        **kwargs,
    ):
        super().__init__(
            env = env,
            policy = None,
            baseline=baseline,
            **kwargs,
        )
        self.env_r = env_r
        self.num_loc_r = env_r.generator.num_loc
        self.policy_d = get_policy(
            policy_d_name, 
            env_name = env.name,
            **policy_d_kwargs
        )
        self.policy_r = get_policy(
            policy_r_name, 
            env_name = env_r.name,
            **policy_r_kwargs
        )
        self.alpha_train = alpha_train
        self.beta_train = beta_train 
        self.alpha_val = alpha_val 
        self.beta_val = beta_val
        self.iters = iters
        self.sub_norm = sub_norm,
        self.w = w
        self.num_loc_min = num_loc_min
        self.num_loc_max = num_loc_max
        self.automatic_optimization = False # enable manual optimization

    def configure_optimizers(self, parameters=None):
        """
        Args:
            parameters: parameters to be optimized. If None, will use `self.parameters()`, i.e. all parameters
        """

        opt_d, sch_d = super().configure_optimizers(self.policy_d.parameters())
        opt_r, sch_r = super().configure_optimizers(self.policy_r.parameters())
        return opt_d + opt_r, [sch_d, sch_r]
    
    def shared_step(self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None):
        if phase == "train":
            if self.env.name in ["tsp", "op", "pctsp"]:
                part_size = self.num_loc_r - 1
            elif self.env.name == "cvrp":
                part_size = self.num_loc_r
            assert self.num_loc_min % part_size == 0
            assert self.num_loc_max % part_size == 0

            if self.env.name == "tsp" and self.current_epoch < 0:
                self.num_loc = self.num_loc_min
                out = self.infer(batch, phase, phase, self.alpha_train * 2, self.beta_train)
            else:
                self.num_loc = torch.randint(self.num_loc_min // part_size, self.num_loc_max // part_size + 1, []).item() * part_size
                out = self.infer(batch, phase, phase, self.alpha_train, self.beta_train)
        else:
            out = self.infer(batch, phase, phase, self.alpha_val, self.beta_val)
        self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
    
    def infer(self, batch: TensorDict, phase_d: str, phase_r: str, ns_d: int, ns_r: int, **kwargs):
        td = self.env.reset(batch)
        if "train" in [phase_d, phase_r] and hasattr(self, "num_loc"):
            td["action_mask"] = td["action_mask"][:, :self.num_loc]
            td["locs"] = td["locs"][:, :self.num_loc, :]
        out_d = self.policy_d.forward(
            td.clone(), 
            env=self.env, 
            phase=phase_d, 
            return_actions=True, 
            num_starts=ns_d, 
            multisample=True
        )

        td = batchify(td, ns_d)
        actions = out_d["actions"]
        for i in range(self.iters):
            if i == 0:
                offset = 0
            elif i == 1:
                offset = self.num_loc_r // 2
            else:
                offset = torch.randint(0, self.num_loc_r, []).item()
            actions, loss = self.revise(
                td = td,
                actions = actions,
                phase = phase_r,
                num_starts = ns_r,
                offset = offset,
            )
            if phase_r == "train":
                # loss_r = loss_r + loss
                opt_d, opt_r = self.optimizers()
                opt_r.zero_grad()
                self.manual_backward(loss)
                opt_r.step()
                out_d.update({f"loss_r{i}": loss})

        reward = unbatchify(self.env.get_reward(td, actions), ns_d)
        par_reward = unbatchify(out_d["reward"], ns_d)
        actions = unbatchify(actions, ns_d)
        out_d.update({"reward": reward, "actions": actions})
        if phase_d == "train":
            ll_d = unbatchify(out_d["log_likelihood"], ns_d)
            self.calculate_loss(td, td, out_d, self.w * reward + (1 - self.w) * par_reward, ll_d)
            opt_d, opt_r = self.optimizers()
            opt_d.zero_grad()
            self.manual_backward(out_d["loss"])
            opt_d.step()
        if ns_d > 1:
            max_reward, max_idxs = reward.max(dim=-1)
            out_d.update({"max_reward": max_reward})
        return out_d

    def revise(
        self, 
        td: TensorDict, 
        actions: torch.Tensor, 
        phase: str,
        num_starts: int,
        offset: int = 0, 
    ):
        device = td.device

        action_utils = get_action_utils(self.env.name)

        # Based on partition actions to get partitions
        td_p, partition, tail, forced_actions = action_utils.partition(td, actions, self.num_loc_r, offset)   
        
        # Batchify the forced_actions according to num_starts
        batch_size, n_partitions, n_nodes = partition.shape
        dirs = forced_actions.size(0) // (batch_size*n_partitions)
        assert dirs in [1, 2]
        assert num_starts % dirs == 0
        forced_actions = batchify(forced_actions, num_starts // dirs)
        
        # set sub environments
        td = self.env_r.reset(batch_size=batch_size*n_partitions).to(device)
        td.update(td_p)

        # Call reviser to solve the sub-routes
        out_r = self.policy_r.forward(
            td.clone(), 
            env = self.env_r,
            phase = phase, 
            return_actions = True,
            actions = forced_actions,
            multisample = True, 
            num_starts = num_starts,  
        )  
        reward = unbatchify(out_r["reward"], num_starts)
        actions = unbatchify(out_r["actions"], num_starts)
        actions = action_utils.process_partial(actions)
        log_likelihood = unbatchify(out_r["log_likelihood"], num_starts)
        if phase == "train":
            out_r = self.calculate_loss(td, td, out_r, reward, log_likelihood)

        # get best actions
        reward, max_idxs = reward.max(dim=1)  
        actions = gather_by_index(actions, max_idxs, dim=1)
        if dirs == 2:
            reverse_flag = max_idxs % 2 == 1
            actions[reverse_flag] = torch.flip(actions[reverse_flag], [-1])
        log_likelihood = gather_by_index(log_likelihood, max_idxs, dim=1)
        
        loss = out_r.get("loss", None)

        # Old reward
        best_revised_actions = torch.arange(n_nodes, device=device)[None, :].expand(batch_size*n_partitions, -1)
        best_revised_reward = self.env_r.get_reward(td, best_revised_actions)
        # Replace with better reward
        improve_flag = reward > best_revised_reward
        best_revised_reward = torch.where(improve_flag, reward, best_revised_reward)
        best_revised_actions = torch.where(improve_flag.unsqueeze(1), actions, best_revised_actions)
        
        # reconnect
        best_revised_actions = rearrange(best_revised_actions, "(b p) l -> b p l", b=batch_size, p=n_partitions)
        best_revised_actions = gather_by_index(partition, best_revised_actions, dim=-1)    
        final_actions = action_utils.reconnect(best_revised_actions, tail)
        return final_actions, loss
        