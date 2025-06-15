# manually import 3rd party classes to register
from src.infra.registry import MODULE_REGISTRY

for cls in []:
    MODULE_REGISTRY.add(cls.__name__, cls)

