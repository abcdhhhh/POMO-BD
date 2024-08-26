from typing import Optional, Union

import torch
import torch.nn as nn
from rl4co.envs import RL4COEnvBase, get_env
from rl4co.models.rl.common.base import RL4COLitModule

from einops import rearrange
from rl4co.models.common.constructive.nonautoregressive import (
    NonAutoregressiveEncoder,
    NonAutoregressiveDecoder,
    NonAutoregressivePolicy,
)
from rl4co.models.zoo.nargnn.encoder import NARGNNEncoder

from rl4co.utils.ops import batchify, gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

log = get_pylogger(__name__)

class UDCPolicy(NonAutoregressivePolicy):
    def __init__(
        self,
        encoder: NonAutoregressiveEncoder = None,
        decoder: NonAutoregressiveDecoder = None,
        env_name: Union[str, RL4COEnvBase] = "tsp",
        reviser: nn.Module = None,
        sub_norm: bool = True,
        iters: int = 2,
        encoder_kwargs = {},
        **kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(
                env_name=env_name,
                **encoder_kwargs,
            )
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            **kwargs,
        )

        self.reviser = reviser
        self.sub_num_loc = reviser.env.generator.num_loc
        self.sub_norm = sub_norm
        self.iters = iters

    def forward(
        self,
        td: TensorDict,
        env: Union[str, RL4COEnvBase, None] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        par_out = super().forward(
            td = td,
            env = env,
            phase = phase,
            calc_reward = False, # We don't need the partition reward
            return_actions = True, # Used for partition
            return_entropy = return_entropy,
            return_init_embeds = return_init_embeds,
            return_sum_log_likelihood = return_sum_log_likelihood,
            actions = actions,
            **decoding_kwargs,
        )

        num_starts = decoding_kwargs.get('num_starts')
        td_sample = batchify(td, num_starts)
        par_actions = par_out["actions"]

        reviser_loss = 0.
        actions = par_actions
        for i in range(self.iters):
            if i == 0:
                offset = 0
            elif i == 1:
                offset = self.sub_num_loc // 2
            else:
                offset = torch.randint(0, self.sub_num_loc, (1,)).item()
            actions, loss = self.revise(td_sample, actions, offset=offset, phase=phase)
            if loss is not None:
                reviser_loss = reviser_loss + loss 
        out = {"reviser_loss": reviser_loss}
        
        if return_actions or calc_reward:
            out["actions"] = actions
            if calc_reward:
                reward = env.get_reward(td_sample, actions)
                out["reward"] = reward
        
        if return_entropy:
            out["entropy"] = par_out["entropy"]
        if return_init_embeds:
            out["init_embeds"] = par_out["init_embeds"]
        if return_sum_log_likelihood:
            out["log_likelihood"] = par_out["log_likelihood"]

        return out
    
    def revise(
        self, 
        td: TensorDict, 
        actions: torch.Tensor, 
        offset: int=0, 
        phase: str="train",
    ):
        device = td.device

        # Based on partition actions to get partitions
        sub_locs, partition, tail = self.partition(td, actions, offset)

        # Batchify the sub_locs along the partitions
        batch_size, n_partitions, n_nodes, _ = sub_locs.shape
        assert n_nodes == self.sub_num_loc
        sub_locs = rearrange(sub_locs, "b p n d -> (b p) n d", b=batch_size, p=n_partitions, n=n_nodes, d=2)

        # set sub environments
        sub_env = self.reviser.env
        sub_td = sub_env.reset(batch_size=batch_size*n_partitions).to(device)
        sub_td.set("locs", sub_locs)

        # Call reviser to solve the sub-routes and record the best
        best_revised_actions = torch.arange(n_nodes, device=device)[None, :].expand(batch_size*n_partitions, -1)
        best_revised_reward = sub_env.get_reward(sub_td, best_revised_actions)
        reviser = self.reviser.to(device)
        sub_num_starts = reviser.num_starts
        import pdb; pdb.set_trace()
        reviser_out = reviser(
            sub_td.clone(), 
            phase=phase, 
            multisample=True, 
            num_starts=sub_num_starts,  
            return_actions=True,
        )

        # get best actions
        if sub_num_starts > 1:
            reward = unbatchify(reviser_out["reward"], sub_num_starts)
            actions = unbatchify(reviser_out["actions"], sub_num_starts)
            log_likelihood = unbatchify(reviser_out["log_likelihood"], sub_num_starts)
            if phase == "train":
                reviser_out = reviser.calculate_loss(sub_td, sub_td, reviser_out, reward, log_likelihood)
            reward, max_idxs = reward.max(dim=1)
            actions = gather_by_index(actions, max_idxs, dim=1)
            log_likelihood = gather_by_index(log_likelihood, max_idxs, dim=1)
        else:
            reward = reviser_out["reward"]
            actions = reviser_out["actions"]
            assert phase != "train"
        loss = reviser_out.get("loss", None)

        # Record the best
        improve_flag = reward > best_revised_reward
        best_revised_reward = torch.where(improve_flag, reward, best_revised_reward)
        best_revised_actions = torch.where(improve_flag.unsqueeze(1), actions, best_revised_actions)

        # reconnect
        best_revised_actions = rearrange(best_revised_actions, "(b p) l -> b p l", b=batch_size, p=n_partitions)
        best_revised_actions = gather_by_index(partition, best_revised_actions, dim=-1)    
        final_actions = self.reconnect(best_revised_actions, tail)
        return final_actions, loss

    def partition(self, td: TensorDict, actions: torch.Tensor, offset: int=0):
        if self.env_name == "tsp":
            locs, partition, tail = self.partition_tsp(td, actions, self.sub_num_loc - 1, offset)
        # normalize sub-graph locs
        if self.sub_norm:
            locs = self.min_max_normalize(locs) 
        return locs, partition, tail
    
    def reconnect(self, actions: torch.Tensor, tail: Optional[torch.Tensor]) -> torch.Tensor:
        if self.env_name == "tsp":
            actions = self.reconnect_tsp(actions, tail)
        return actions
    
    @staticmethod
    def partition_tsp(td: TensorDict, actions: torch.Tensor, n_edges: int, offset: int=0):
        """
        Args:
            td [bs*n_samples]
            actions [bs*n_samples, seq_len]
            n_edges = n_nodes-1
        Returns:
            locs [bs*n_samples, n_partitions, n_nodes, 2]
            partition [bs*n_samples, n_partitions, n_nodes]
            tail [bs*n_samples, r] or None
        """
        seq_len = actions.size(-1)
        n_partitions = seq_len // n_edges
        par_len = n_partitions * n_edges
        actions = actions.roll(shifts=-offset, dims=1)
        partition = actions[:, :par_len].reshape(-1, n_partitions, n_edges)    
        if seq_len > par_len:
            tail = actions[:, par_len:]
            last_node = actions[:, n_edges::n_edges, None]
        else:
            tail = None 
            last_node = actions[:, 0::n_edges, None].roll(shifts=-1, dims=1)
        partition = torch.cat(
            [
                partition,
                last_node,
            ],
            dim=-1,
        )
        # Expand the locs
        locs = td["locs"].unsqueeze(1).expand(-1, n_partitions, -1, -1)
        # Get the locations of the partitions
        locs = gather_by_index(locs, partition, dim=-2)

        return locs, partition, tail

    @staticmethod
    def reconnect_tsp(actions: torch.Tensor, tail: Optional[torch.Tensor], offset: int=0):
        """
        Args:
            actions [bs*n_samples, n_partitions, n_nodes]
            tail [bs*n_samples, r] or None
        Returns:
            actions [bs*n_samples, seq_len]
        """
        actions = actions[:, :, :-1].flatten(start_dim=1)
        if tail is not None:
            actions = torch.cat(
                [
                    actions,
                    tail,
                ],
                dim=-1,
            )
        return actions
    
    @staticmethod
    def min_max_normalize(locs: torch.Tensor):
        """
        Args:
            locs [..., n, 2]
        """
        locs_min = locs.min(dim=-2, keepdim=True).values
        locs_max = locs.max(dim=-2, keepdim=True).values
        locs_scale = (locs_max - locs_min).max(dim=-1, keepdim=True).values
        return (locs - locs_min) / locs_scale
