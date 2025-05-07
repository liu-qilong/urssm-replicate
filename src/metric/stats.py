import torch
from torch import nn

from src.tool.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
class MeanDiffRatio(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return (y_pred.mean() - y_true.mean()) / y_true.mean()
    

@METRIC_REGISTRY.register()
class StdDiffRatio(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return (y_pred.std() - y_true.std()) / y_true.std()