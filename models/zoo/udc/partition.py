from typing import Optional, Union

import torch 
from tensordict import TensorDict

from rl4co.utils.ops import batchify, gather_by_index, unbatchify

def partition_tsp(td: TensorDict, actions: torch.Tensor, n_nodes: int, offset: int=0, sub_norm: bool=True):
    """
    Args:
        td [B]:
            locs [B, N, 2]
        actions [B, L]
        n_edges = n - 1
    Returns:
        out [B*p]:
            locs [B*p, n, 2]
        partition [B, p, n]
        tail [B, r] or None
        L = p * (n - 1) + r
    """
    L = actions.size(-1)
    n_edges = n_nodes - 1
    n_partitions = L // n_edges
    L0 = n_partitions * n_edges
    # Roll actions
    actions = actions.roll(shifts=-offset, dims=1)

    # Get partition and tail
    partition = actions[:, :L0].reshape(-1, n_partitions, n_edges)    
    if L > L0:
        tail = actions[:, L0:]
        last_node = actions[:, n_edges::n_edges, None]
    else:
        tail = None 
        last_node = actions[:, 0::n_edges, None].roll(shifts=-1, dims=1)
    partition = torch.cat([partition, last_node], dim=-1)

    # Get the locations of the partitions
    locs = gather_by_index(td["locs"], partition.flatten(-2), dim=-2) # [B, p*n, 2]
    locs = locs.reshape(-1, n_nodes, 2)

    # normalize
    if sub_norm:
        locs = min_max_normalize(locs)
    td_p = TensorDict({"locs": locs})
    return td_p, partition, tail

def reconnect_tsp(actions: torch.Tensor, tail: Optional[torch.Tensor], offset: int=0):
    """
    Args:
        actions [B, p, n]
        tail [B, r] or None
    Returns:
        actions [B, L]
        L = p * (n - 1) + r
    """
    actions = actions[:, :, :-1].flatten(start_dim=1)
    if tail is not None:
        actions = torch.cat([actions, tail], dim=-1)
    return actions

def partition_cvrp(td: TensorDict, actions: torch.Tensor, n_edges: int, offset: int=0):
    """
    Args:
        td [B]:
            locs [B, N, 2]
            capacity [B, 1]
            demand [B, N-1]
            depot [B, 2]
        actions [B, L]
        n_edges = n - 1
    Returns:
        locs [B, p, n, 2]
        partition [B, p, n]
        tail [B, r] or None
        L = p * n_edges + r
    """
    pass

def reconnect_cvrp(actions: torch.Tensor, tail: Optional[torch.Tensor], offset: int=0):
    pass

def partition_pctsp(td: TensorDict, actions: torch.Tensor, n_edges: int, offset: int=0):
    pass

def reconnect_pctsp(actions: torch.Tensor, tail: Optional[torch.Tensor], offset: int=0):
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

def reverse_aug(td: TensorDict, partition: torch.Tensor):
    """
    Args:
        td:
            e.g.
            locs [B*p, n, 2]
        partition [B, p, n]
    Returns:
        bitd:
            e.g.
            locs [2*B*p, n, 2]
        bipartition [2*B, p, n]
    """
    bitd = td.clone()
    for k, v in bitd.items():
        bitd[k] = torch.cat([v, torch.flip(v, [-2])], dim=0)
    bipartition = torch.cat([partition, torch.flip(partition, [-1])], dim=0)
    return bitd, bipartition