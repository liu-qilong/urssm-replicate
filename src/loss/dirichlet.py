# This implementation is adapted from Dongliang Cao, et al. (2024): https://github.com/dongliangcao/unsupervised-learning-of-robust-spectral-shape-matching. New modules & refinements are added

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.infra.registry import LOSS_REGISTRY


def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)


@LOSS_REGISTRY.register()
class DirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, L):
        assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, torch.bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return self.loss_weight * loss
