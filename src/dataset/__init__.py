from . import static2dynamic, footprint2pressure

from src.tool.registry import DATASET_REGISTRY
from torch.utils.data import Dataset

for cls in [Dataset, ]:
    DATASET_REGISTRY.add(cls.__name__, cls)