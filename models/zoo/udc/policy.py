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
        return_partitions: bool = False,
        return_partitions_actions: bool = False,
        actions=None,
        **decoding_kwargs,
    ) -> dict:
        device = td.device
        
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
        par_log_likelihood = par_out["log_likelihood"]

        # Based on partition actions to get partitions
        sub_locs, partition, tail = self.partition(td_sample, par_actions)

        # Batchify the shpp_td along the partitions
        batch_size = sub_locs.size(0)
        n_partitions = sub_locs.size(1)
        n_nodes = sub_locs.size(2)
        assert self.sub_num_loc == n_nodes
        sub_locs = rearrange(sub_locs, "b p n d -> (b p) n d", b=batch_size, p=n_partitions, n=n_nodes, d=2)
        
        # set sub environments
        sub_td = self.reviser.env.reset(batch_size=batch_size*n_partitions).to(device)
        sub_td.set("locs", sub_locs)

        # Call reviser to solve the sub-routes and record the best
        best_revised_reward = torch.full(sub_td.shape[:1], float("-inf")).to(device)
        best_revised_actions = torch.zeros(sub_td["locs"].shape[:-1], dtype=torch.int64).to(device)
        reviser = self.reviser.to(device)
        sub_num_starts = reviser.num_starts
        reviser_out = reviser(
            sub_td.clone(), 
            phase=phase, 
            multisample=True, 
            num_starts=sub_num_starts,  
            return_actions=True,
        )
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

        # Record the best
        improve_flag = reward > best_revised_reward
        best_revised_reward = torch.where(improve_flag, reward, best_revised_reward)
        best_revised_actions = torch.where(improve_flag.unsqueeze(1), actions, best_revised_actions)

        # Construct final output
        # log_likelihood = rearrange(log_likelihood, "(b p) -> b p", b=batch_size, p=n_partitions)
        out = {
            "log_likelihood": par_log_likelihood,
            "reviser_loss": reviser_out.get("loss", None)
        }
        
        if return_actions or calc_reward:
            final_actions = rearrange(best_revised_actions, "(b p) l -> b p l", b=batch_size, p=n_partitions)
            final_actions = gather_by_index(partition, final_actions, dim=-1)    
            final_actions = self.reconnect(final_actions, tail)
            out["actions"] = final_actions
            if calc_reward:
                final_reward = env.get_reward(td_sample, final_actions)
                out["reward"] = final_reward
        
        if return_entropy:
            out["entropy"] = par_out["entropy"]
        if return_init_embeds:
            out["init_embeds"] = par_out["init_embeds"]
        if return_partitions:
            out["partition"] = partition
        if return_partitions_actions:
            out["par_actions"] = par_actions
            out["revised_actions"] = best_revised_actions

        return out

    def partition(self, td: TensorDict, actions: torch.Tensor):
        if self.env_name in ["tsp", "pctsp"]:
            locs, partition, tail = self.partition_tsp(td, actions, self.sub_num_loc - 1)
        if self.sub_norm:
            locs = self.min_max_normalize(locs) 
        return locs, partition, tail
    
    def reconnect(self, actions: torch.Tensor, tail: Optional[torch.Tensor]):
        if self.env_name in ["tsp", "pctsp"]:
            actions = self.reconnect_tsp(actions, tail)
        return actions
    
    @staticmethod
    def partition_tsp(td: TensorDict, actions: torch.Tensor, n_edges: int):
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
        partition = actions[:, :par_len].reshape(-1, n_partitions, n_edges)
        if seq_len > par_len:
            tail = actions[:, par_len:]
            last_node = actions[:, n_edges::n_edges, None]
        else:
            tail = None 
            last_node = actions[:, 0::n_edges, None].roll(shifts=-1,dims=1)
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
    def reconnect_tsp(actions: torch.Tensor, tail: Optional[torch.Tensor]):
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
