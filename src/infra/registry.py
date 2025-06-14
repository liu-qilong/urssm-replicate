class Registry(dict):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(Registry, self).__init__(*args, **kwargs)

    def add(self, name, obj):
        assert (name not in self), f"An object named '{name}' was already registered in {self.name} registry"

        self[name] = obj

    def get(self, name):
        return self[name]

    def register(self):
        """decorator factory returns a decorator function"""
        def register_fn(fn):
            self.add(fn.__name__, fn)
            return fn

        return register_fn
    

DATASET_REGISTRY = Registry('dataset')
DATALOADER_REGISTRY = Registry('dataloader')
NETWORK_REGISTRY = Registry('network')
MODULE_REGISTRY = Registry('module')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
OPTIMIZER_REGISTRY = Registry('optimizer')
SCRIPT_REGISTRY = Registry('script')