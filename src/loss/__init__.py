from src.tool.registry import LOSS_REGISTRY
from torch.nn import MSELoss

for cls in [MSELoss, ]:
    LOSS_REGISTRY.add(cls.__name__, cls)