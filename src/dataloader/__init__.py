from src.tool.registry import DATALOADER_REGISTRY
from torch.utils.data import DataLoader

for cls in [DataLoader, ]:
    DATALOADER_REGISTRY.add(cls.__name__, cls)