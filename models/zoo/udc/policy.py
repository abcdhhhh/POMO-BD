from typing import Optional, Union

import torch
import torch.nn as nn
from rl4co.envs import RL4COEnvBase, get_env

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
        n_samples: int = 40,
        reviser: nn.Module = None,
        **encoder_kwargs,
    ):
        if encoder is None:
            encoder = NARGNNEncoder(**encoder_kwargs)
        if decoder is None:
            decoder = NonAutoregressiveDecoder()

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            train_decode_type="multistart_sampling",
            val_decode_type="multistart_sampling",
            test_decode_type="multistart_sampling",
        )

        self.n_samples = n_samples
        self.reviser = reviser

    @staticmethod
    def partition(td: TensorDict, actions: torch.Tensor):
        """
        TODO
        Args:
            td [bs*n_samples]
            actions [bs*n_samples, seq_len]
        Returns:
            <TensorDict>
                locs [bs*n_samples, n_partitions, n_nodes, 2]
                partition [bs*n_samples, n_partitions, seq_len]
        """
        max_num_partitions = 0
        max_len_sequence = 0
        partition = torch.zeros([*actions.size(), actions.size(-1)]).to(
            td.device, torch.int64
        )  # [bs*n_samples, seq_len, seq_len]
        for batch_idx in range(td.size(0)):
            partition_idx = 0
            partition_start_idx = 0
            for action_idx, action in enumerate(actions[batch_idx]):
                if (action == 0) & (action_idx != 0):
                    partition_idx += 1
                    # Update the max length of the sequence
                    if action_idx - partition_start_idx > max_len_sequence:
                        max_len_sequence = action_idx - partition_start_idx
                    partition_start_idx = action_idx + 1
                else:
                    partition[
                        batch_idx, partition_idx, action_idx - partition_start_idx
                    ] = action
            # Update the max number of partitions
            if partition_idx + 1 > max_num_partitions:
                max_num_partitions = partition_idx + 1
        # Squeese the partition
        partition = partition[:, :max_num_partitions, :max_len_sequence]
        # Adding depot to the beginning and the end
        partition = torch.cat(
            [
                torch.zeros_like(partition[:, :, :1]),
                partition,
                torch.zeros_like(partition[:, :, :1]),
            ],
            dim=-1,
        )
        # Expand the locs
        locs = td["locs"].unsqueeze(1).expand(-1, max_num_partitions, -1, -1)
        # Get the locations of the partitions
        locs = gather_by_index(locs, partition, dim=-2)

        return locs, partition
