from src.tool.registry import METRIC_REGISTRY
from torch.nn import L1Loss, MSELoss

from . import stats, sample

for cls in [L1Loss, MSELoss]:
    METRIC_REGISTRY.add(cls.__name__, cls)