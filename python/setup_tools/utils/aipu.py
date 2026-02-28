def precompile_hock(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends_list = [*default_backends, "flir"]
    kargs["default_backends"] = tuple(default_backends_list)
    default_backends = tuple(default_backends_list)
    return default_backends
