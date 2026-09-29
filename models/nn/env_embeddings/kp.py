import torch
import torch.nn as nn

from rl4co.envs.common.utils import batch_to_scalar
from rl4co.utils.ops import gather_by_index
from rl4co.models.nn.env_embeddings.context import EnvContext, VRPContext
from rl4co.models.nn.env_embeddings.init import TSPInitEmbedding

class KPInitEmbedding(TSPInitEmbedding):
    pass

class KPContext(EnvContext):
    def __init__(self, embed_dim):
        super(KPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )

    def _state_embedding(self, embeddings, td):
        state_embedding = td["capacity"]
        return state_embedding


    