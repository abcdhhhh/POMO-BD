from typing import Optional, Union

import torch 
from tensordict import TensorDict

from rl4co.utils.ops import batchify, gather_by_index, unbatchify

def partition_tsp(td: TensorDict, actions: torch.Tensor, n_edges: int, offset: int=0):
    """
    Args:
        td [bs*n_samples]:
            locs [bs*n_samples, n_nodes, 2]
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
    actions = actions.roll(shifts=offset, dims=1)
    return actions