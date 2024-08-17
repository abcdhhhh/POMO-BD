import torch
import torch.nn as nn
from rl4co.models.nn.env_embeddings.context import EnvContext 
from rl4co.utils.ops import gather_by_index

class SHPPContext(EnvContext):
    """Context embedding for the Shortest Hamiltonian Path Problem (SHPP)
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
        - terminating node embedding
    """

    def __init__(self, embed_dim):
        super().__init__(embed_dim, 3 * embed_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(3 * self.embed_dim).uniform_(-1, 1)
        )

    def forward(self, embeds, td):
        batch_size = embeds.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            if len(td.batch_size) < 2:
                context_embedding = self.W_placeholder[None, :].expand(
                    batch_size, self.W_placeholder.size(-1)
                )
            else:
                context_embedding = self.W_placeholder[None, None, :].expand(
                    batch_size, td.batch_size[1], self.W_placeholder.size(-1)
                )
        else:
            context_embedding = gather_by_index(
                embeds,
                torch.stack(
                    [td["first_node"], td["current_node"], td["last_node"]], -1
                ).view(batch_size, -1),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)