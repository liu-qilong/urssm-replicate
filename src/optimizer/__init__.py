# manually import 3rd party classes to register
from src.tool.registry import OPTIMIZER_REGISTRY
from torch.optim import SGD

for cls in [SGD, ]:
    OPTIMIZER_REGISTRY.add(cls.__name__, cls)