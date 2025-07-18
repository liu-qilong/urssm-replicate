# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.infra.registry import MODULE_REGISTRY


@MODULE_REGISTRY.register()
class SoftmaxPermutationMatrix(nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super().__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, feat_x, feat_y, verts_mask_x, verts_mask_y):
        log_alpha = torch.bmm(
            F.normalize(feat_x, dim=-1, p=2),
            F.normalize(feat_y, dim=-1, p=2).transpose(1, 2),
        ) / self.tau

        # softmax
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))

        if self.hard:
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            # straight through estimator
            ret = alpha_hard - alpha.detach() + alpha
        
        else:
            ret = alpha

        return ret * verts_mask_x.unsqueeze(-1) * verts_mask_y.unsqueeze(-2)  # mask out padded points [B, V_x, V_y] * [B, V_x, 1] * [B, 1, V_y]