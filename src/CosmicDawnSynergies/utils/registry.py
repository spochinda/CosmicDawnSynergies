class Registry:
    """Name -> class registry (basicsr-style).

    Components register themselves with a decorator:

        MODEL_REGISTRY = Registry('model')

        @MODEL_REGISTRY.register()
        class MLPModel: ...

    and are resolved from yml ``type:`` strings with ``MODEL_REGISTRY.get(name)``.
    Registration happens on import; each component subpackage auto-imports its
    ``*_arch.py`` / ``*_model.py`` / ``*_dataset.py`` modules in ``__init__.py``.
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        if name in self._obj_map:
            raise KeyError(f"'{name}' already registered in '{self._name}' registry")
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            def deco(cls):
                self._do_register(cls.__name__, cls)
                return cls
            return deco
        self._do_register(obj.__name__, obj)
        return obj

    def get(self, name):
        obj = self._obj_map.get(name)
        if obj is None:
            raise KeyError(f"'{name}' not found in '{self._name}' registry. "
                           f"Available: {list(self._obj_map)}")
        return obj

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
DATA_REGISTRY = Registry('data')
LIKELIHOOD_REGISTRY = Registry('likelihood')
