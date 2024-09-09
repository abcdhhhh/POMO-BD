from typing import Optional, Union

import torch 
from tensordict import TensorDict

from rl4co.utils.ops import batchify, gather_by_index, unbatchify

def get_action_utils(env_name: str):
    if env_name == "tsp":
        return TSPActionUtils()
    elif env_name == "cvrp":
        return CVRPActionUtils()
    else:
        raise NotImplementedError

class ActionUtils:
    """
    Give some static-methods for divide-and-conquer actions
    """
    @staticmethod
    def partition(td: TensorDict, actions: torch.Tensor, num_loc_r: int, offset: int=0, sub_norm: bool=True):
        raise NotImplementedError
    
    @staticmethod
    def process_partial(actions: torch.Tensor):
        return actions
    
    @staticmethod
    def reconnect(actions: torch.Tensor, tail: torch.Tensor=None):
        raise NotImplementedError
    

class TSPActionUtils(ActionUtils):
    @staticmethod
    def partition(td: TensorDict, actions: torch.Tensor, num_loc_r: int, offset: int=0, sub_norm: bool=True):
        """
        Args:
            td [B]:
                locs [B, N, 2]
            actions [B, L]
        Returns:
            td_p [B*p]:
                locs [B*p, n, 2]
            partition [B, p, n]
            tail [B, r] or None    L = p * (n - 1) + r
            forced_actions [2*B*p, 2]
        """
        batch_size, n_actions = actions.shape
        n_edges = num_loc_r - 1
        n_partitions = n_actions // n_edges
        n_rest = n_actions % n_edges
        # Roll actions
        actions = actions.roll(shifts=-offset, dims=1)

        # Get partition and tail
        partition = actions[:, :n_actions-n_rest].reshape(-1, n_partitions, n_edges)    
        if n_rest:
            assert False
            tail = actions[:, n_actions-n_rest:]
            last_node = actions[:, n_edges::n_edges, None]
        else:
            tail = None 
            last_node = actions[:, 0::n_edges, None].roll(shifts=-1, dims=1)
        partition = torch.cat([partition, last_node], dim=-1)

        # Get the locations of the partitions
        locs = gather_by_index(td["locs"], partition.flatten(-2), dim=-2) # [B, p*n, 2]
        locs = locs.reshape(-1, num_loc_r, 2)

        # Normalize
        if sub_norm:
            locs = min_max_normalize(locs)
        td_p = TensorDict({"locs": locs})
        # 2 directions
        forced_actions = torch.tensor(
                [[[num_loc_r - 1, 0]], [[0, num_loc_r - 1]]], 
                device=td.device,
            ).expand(2, batch_size * n_partitions, 2).flatten(0, 1)

        return td_p, partition, tail, forced_actions
    
    @staticmethod
    def process_partial(actions: torch.Tensor):
        """
        Args:
            actions [..., L]
        Returns:
            actions [..., L]
        """
        return actions.roll(shifts=-1, dims=-1)
    
    @staticmethod
    def reconnect(actions: torch.Tensor, tail: torch.Tensor=None):
        """
        Args:
            actions [B, p, n]
            tail [B, r] or None
        Returns:
            actions [B, L]
            L = p * (n - 1) + r
        """
        actions = actions[:, :, :-1].flatten(1, 2)
        if tail is not None:
            actions = torch.cat([actions, tail], dim=-1)
        return actions

class CVRPActionUtils(ActionUtils):
    @staticmethod
    def partition(td: TensorDict, actions: torch.Tensor, num_loc_r: int, offset: int=0, sub_norm: bool=True): 
        """
        Args:
            td [B]:
                locs [B, N, 2]
                capacity [B, 1]
                demand [B, N-1]
                depot [B, 2]
            actions [B, L]
        Returns:
            td_p: [B*p]
                locs [B*p, n, 2]
            partition [B, p, n]
        """
        pass

def min_max_normalize(locs: torch.Tensor):
    """
    Args:
        locs [..., n, 2]
    """
    locs_min = locs.min(dim=-2, keepdim=True).values
    locs_max = locs.max(dim=-2, keepdim=True).values
    locs_scale = (locs_max - locs_min).max(dim=-1, keepdim=True).values
    return (locs - locs_min) / locs_scale
