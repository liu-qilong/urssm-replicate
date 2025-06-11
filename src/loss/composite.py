import torch
import torch.nn as nn

from src.infra.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CompositeLoss(nn.Module):
    def __init__(self, opt: dict):
        super(CompositeLoss, self).__init__()
        self.opt = opt
        self.losses = nn.ModuleList()
        self.weights = []

        for loss in self.opt.train.loss.losses:
            loss_name = loss['name']
            loss_args = loss['args']
            loss_weight = loss['weight']
            self.losses.append(LOSS_REGISTRY[loss_name](**loss_args))
            self.weights.append(loss_weight)

    def forward(self, *args, **kwargs):
        total_loss = 0.0

        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(*args, **kwargs)
        
        return total_loss