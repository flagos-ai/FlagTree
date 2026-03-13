def precompile_hook(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends_list = list(default_backends)
    default_backends_list.append('triton_shared')
    kargs["default_backends"] = tuple(default_backends_list)
    default_backends = tuple(default_backends_list)
    return default_backends
