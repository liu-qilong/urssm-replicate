# manually import 3rd party classes to register
from src.infra.registry import DATALOADER_REGISTRY
from torch.utils.data import DataLoader

for cls in [DataLoader, ]:
    DATALOADER_REGISTRY.add(cls.__name__, cls)