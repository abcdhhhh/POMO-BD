from einops import rearrange
import torch
import torch.nn.functional as F

def problem_symmetricity_loss(reward, log_likelihood, dim=1, po_alpha=0., elite_k=1, elite_alpha=0.):
    """REINFORCE loss for problem symmetricity
    Baseline is the average reward for all augmented problems
    Corresponds to `L_ps` in the SymNCO paper
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    if po_alpha != 0:
        assert dim == 1
        preference = reward[:, :, None, ...] > reward[:, None, :, ...] 
        log_prob_pair = log_likelihood[:, :, None, ...] - log_likelihood[:, None, :, ...] 
        pf_log = torch.log(F.sigmoid(po_alpha * log_prob_pair)) 
        loss = -(pf_log * preference).mean()
    else:
        advantage = reward - reward.mean(dim=dim, keepdim=True)
        loss = -(advantage * log_likelihood).mean()
        if elite_alpha != 0:
            top_reward, top_idx = reward.topk(elite_k, dim=dim)
            top_loss = -(advantage * log_likelihood).gather(1, top_idx).mean()
            loss = elite_alpha * top_loss + (1 - elite_alpha) * loss
    return loss


def solution_symmetricity_loss(reward, log_likelihood, dim=-1, po_alpha=0., elite_k=1, elite_alpha=0.):
    """REINFORCE loss for solution symmetricity
    Baseline is the average reward for all start nodes
    Corresponds to `L_ss` in the SymNCO paper
    """
    num_augment = reward.shape[dim]
    if num_augment < 2:
        return 0
    if po_alpha != 0:
        assert dim == -1
        preference = reward[..., :, None] > reward[..., None, :] 
        log_prob_pair = log_likelihood[..., :, None] - log_likelihood[..., None, :] 
        pf_log = torch.log(F.sigmoid(po_alpha * log_prob_pair)) 
        loss = -(pf_log * preference).mean()
    else:
        advantage = reward - reward.mean(dim=dim, keepdim=True)
        loss = -(advantage * log_likelihood).mean()
        if elite_alpha != 0:
            top_reward, top_idx = reward.topk(elite_k, dim=dim)
            top_loss = -(advantage * log_likelihood).gather(1, top_idx).mean()
            loss = elite_alpha * top_loss + (1 - elite_alpha) * loss     
    return loss

def mixed_loss(reward, log_likelihood, po_alpha=0., elite_k=1, elite_alpha=0.):
    return problem_symmetricity_loss(reward.flatten(1), log_likelihood.flatten(1), 1, po_alpha, elite_k, elite_alpha)

def invariance_loss(proj_embed, num_augment):
    """Loss for invariant representation on projected nodes
    Corresponds to `L_inv` in the SymNCO paper
    """
    pe = rearrange(proj_embed, "(b a) ... -> b a ...", a=num_augment)
    similarity = sum(
        [F.cosine_similarity(pe[:, 0], pe[:, i], dim=-1) for i in range(1, num_augment)]
    )
    return similarity.mean()
