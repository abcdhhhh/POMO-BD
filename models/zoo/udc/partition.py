from typing import Optional, Union

import torch 
from tensordict import TensorDict

from rl4co.utils.ops import batchify, gather_by_index, unbatchify

def partition_tsp(td: TensorDict, actions: torch.Tensor, n_edges: int, offset: int=0):
    """
    Args:
        td [B]:
            locs [B, N, 2]
        actions [B, L]
        n_edges = n - 1
    Returns:
        locs [B, p, n, 2]
        partition [B, p, n]
        tail [B, r] or None
        L = p * n_edges + r
    """
    L = actions.size(-1)
    n_parts = L // n_edges
    L0 = n_parts * n_edges
    actions = actions.roll(shifts=-offset, dims=1)
    partition = actions[:, :L0].reshape(-1, n_parts, n_edges)    
    if L > L0:
        tail = actions[:, L0:]
        last_node = actions[:, n_edges::n_edges, None]
    else:
        tail = None 
        last_node = actions[:, 0::n_edges, None].roll(shifts=-1, dims=1)
    partition = torch.cat([partition, last_node], dim=-1)

    # Expand the locs
    locs = td["locs"].unsqueeze(1).expand(-1, n_parts, -1, -1)
    # Get the locations of the partitions
    locs = gather_by_index(locs, partition, dim=-2)

    return locs, partition, tail

def reconnect_tsp(actions: torch.Tensor, tail: Optional[torch.Tensor], offset: int=0):
    """
    Args:
        actions [B, p, n]
        tail [B, r] or None
    Returns:
        actions [B, L]
        L = p * n_edges + r
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
