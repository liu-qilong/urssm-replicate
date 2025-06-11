# manually import 3rd party classes to register
from src.infra.registry import OPTIMIZER_REGISTRY
from torch.optim import SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta

for cls in [SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta]:
    OPTIMIZER_REGISTRY.add(cls.__name__, cls)