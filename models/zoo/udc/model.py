from functools import partial
from typing import Optional, Any

import torch
import torch.nn as nn
from rl4co.models.rl.common.base import RL4COLitModule
from tensordict import TensorDict
from einops import rearrange
from rl4co.utils.ops import gather_by_index, unbatchify, batchify
from .partition import partition_tsp, reconnect_tsp, reverse_aug
from rl4co.utils.optim_helpers import create_optimizer, create_scheduler

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class UDC(RL4COLitModule): 
    def __init__(
        self,
        model_d: RL4COLitModule,    # divider
        model_r: RL4COLitModule,    # reviser
        alpha: int,
        beta: int,
        iters: int = 2,
        sub_norm: bool = True,
        dirs: int = 2,
        w: float = 0.9,
        **kwargs,
    ):
        super().__init__(
            env = model_d.env,
            policy = None,
            **kwargs,
        )
        self.model_d = model_d
        self.model_r = model_r
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.sub_norm = sub_norm,
        self.dirs = dirs
        self.w = w

        self.automatic_optimization = False # enable manual optimization

    def configure_optimizers(self, parameters=None):
        """
        Args:
            parameters: parameters to be optimized. If None, will use `self.parameters()`, i.e. all parameters
        """

        opt_d, sch_d = super().configure_optimizers(self.model_d.parameters())
        opt_r, sch_r = super().configure_optimizers(self.model_r.parameters())
        # optim_conf = [opt_d, opt_r], [sch_d, sch_r]
        # import pdb; pdb.set_trace()
        return opt_d + opt_r, [sch_d, sch_r]
    
    def infer(self, batch: TensorDict, phase_d: str, phase_r: str, ns_d: int, ns_r: int, **kwargs):
        td = self.env.reset(batch)
        out_d = self.model_d.forward(
            td.clone(), phase=phase_d, num_starts=ns_d, multisample=True, return_actions=True
        )

        td = batchify(td, ns_d)
        actions = out_d["actions"]
        for i in range(self.iters):
            if i == 0:
                offset = 0
            elif i == 1:
                offset = self.model_r.env.generator.num_loc // 2
            else:
                offset = torch.randint(0, self.model_r.env.generator.num_loc, (1,)).item()
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

        reward = unbatchify(self.model_d.env.get_reward(td, actions), ns_d)
        par_reward = unbatchify(out_d["reward"], ns_d)
        actions = unbatchify(actions, ns_d)
        out_d.update({"reward": reward, "actions": actions})
        if phase_d == "train":
            ll_d = unbatchify(out_d["log_likelihood"], ns_d)
            self.model_d.calculate_loss(td, td, out_d, self.w * reward + (1 - self.w) * par_reward, ll_d)
            opt_d, opt_r = self.optimizers()
            opt_d.zero_grad()
            self.manual_backward(out_d["loss"])
            opt_d.step()
        if ns_d > 1:
            max_reward, max_idxs = reward.max(dim=-1)
            out_d.update({"max_reward": max_reward})
        return out_d
    
    def shared_step(self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None):
        if phase == "train":
            out = self.infer(batch, phase, phase, self.alpha, self.beta)
        else:
            out = self.infer(batch, phase, phase, self.alpha, 1)
        self.log_metrics(out, phase, dataloader_idx=dataloader_idx)

    def revise(
        self, 
        td: TensorDict, 
        actions: torch.Tensor, 
        phase: str,
        num_starts: int,
        offset: int = 0, 
    ):
        device = td.device

        # Based on partition actions to get partitions
        td_p, partition, tail = self.partition(td, actions, offset)
        if self.dirs == 2:
            td_p, partition = reverse_aug(td_p, partition)

        # Batchify the sub_locs along the partitions
        batch_size, n_partitions, n_nodes = partition.shape
        
        # set sub environments
        td = self.model_r.env.reset(batch_size=batch_size*n_partitions).to(device)
        td.update(td_p)

        # Call reviser to solve the sub-routes and record the best
        out_r = self.model_r.forward(
            td.clone(), 
            phase=phase, 
            multisample=True, 
            num_starts=num_starts,  
            return_actions=True,
        )
        
        # get best actions
        reward = unbatchify(out_r["reward"], num_starts * self.dirs)
        actions = unbatchify(out_r["actions"], num_starts * self.dirs)
        log_likelihood = unbatchify(out_r["log_likelihood"], num_starts * self.dirs)
        if phase == "train":
            out_r = self.model_r.calculate_loss(td, td, out_r, reward, log_likelihood)
        reward, max_idxs = reward.max(dim=1)  

        actions = gather_by_index(actions, max_idxs, dim=1)
        if self.dirs == 2:
            reverse_flag = max_idxs % 2 == 1
            actions[reverse_flag] = torch.flip(n_nodes - 1 - actions[reverse_flag], [-1])
        log_likelihood = gather_by_index(log_likelihood, max_idxs, dim=1)
        
        loss = out_r.get("loss", None)

        # Old reward
        best_revised_actions = torch.arange(n_nodes, device=device)[None, :].expand(batch_size*n_partitions, -1)
        best_revised_reward = self.model_r.env.get_reward(td, best_revised_actions)

        # Restore reversed 
        if self.dirs == 2:
            batch_size = batch_size // self.dirs
            assert torch.allclose(best_revised_reward[:batch_size*n_partitions], best_revised_reward[batch_size*n_partitions:])
            best_revised_actions = best_revised_actions[:batch_size*n_partitions]
            best_revised_reward = best_revised_reward[:batch_size*n_partitions]
            partition = partition[:batch_size]

        # Replace with better reward
        improve_flag = reward > best_revised_reward
        best_revised_reward = torch.where(improve_flag, reward, best_revised_reward)
        best_revised_actions = torch.where(improve_flag.unsqueeze(1), actions, best_revised_actions)

        # reconnect
        best_revised_actions = rearrange(best_revised_actions, "(b p) l -> b p l", b=batch_size, p=n_partitions)
        best_revised_actions = gather_by_index(partition, best_revised_actions, dim=-1)    
        final_actions = self.reconnect(best_revised_actions, tail)
        return final_actions, loss

    def partition(self, td: TensorDict, actions: torch.Tensor, offset: int=0):
        if self.model_d.policy.env_name == "tsp":
            out, partition, tail = partition_tsp(td, actions, self.model_r.env.generator.num_loc, offset, self.sub_norm)
        return out, partition, tail
    
    def reconnect(self, actions: torch.Tensor, tail: Optional[torch.Tensor]) -> torch.Tensor:
        if self.model_d.policy.env_name == "tsp":
            actions = reconnect_tsp(actions, tail)
        return actions
        