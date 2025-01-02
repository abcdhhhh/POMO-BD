from einops import rearrange
import torch
import torch.nn.functional as F


def symmetricity_loss(reward, log_likelihood, dim=1, po_alpha=0., elite_k=1, elite_alpha=0.):
    """REINFORCE loss for solution symmetricity
    dim = 1 for all augmented problems
    dim = -1 for all start nodes
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    if po_alpha != 0:
        if dim == 1:
            preference = reward[:, :, None, ...] > reward[:, None, :, ...] 
            log_prob_pair = log_likelihood[:, :, None, ...] - log_likelihood[:, None, :, ...] 
        elif dim == -1:
            preference = reward[..., :, None] > reward[..., None, :] 
            log_prob_pair = log_likelihood[..., :, None] - log_likelihood[..., None, :]
        else:
            raise NotImplementedError 
        
        pf_log = torch.log(F.sigmoid(po_alpha * log_prob_pair)) 
        loss = -(pf_log * preference).mean()
    else:
        advantage = reward - reward.mean(dim=dim, keepdim=True)
        if elite_alpha:
            # Log-likelihood mask. Mask everything but the best rollout per instance
            best_idx = (-reward).argsort(dim).argsort(dim)
            mask = (best_idx < elite_k)
            mask = mask * elite_alpha * reward.shape[dim] / elite_k + 1
        else:
            mask = torch.ones_like(reward)
        loss = -(advantage * log_likelihood * mask).mean()    
    return loss

def mixed_loss(reward, log_likelihood, po_alpha=0., elite_k=1, elite_alpha=0.):
    return symmetricity_loss(reward.flatten(1), log_likelihood.flatten(1), 1, po_alpha, elite_k, elite_alpha)

def invariance_loss(proj_embed, num_augment):
    """Loss for invariant representation on projected nodes
    Corresponds to `L_inv` in the SymNCO paper
    """
    pe = rearrange(proj_embed, "(b a) ... -> b a ...", a=num_augment)
    similarity = sum(
        [F.cosine_similarity(pe[:, 0], pe[:, i], dim=-1) for i in range(1, num_augment)]
    )
    return similarity.mean()
