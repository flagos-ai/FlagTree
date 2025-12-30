def precompile_hock(*args, **kargs):
    default_backends = kargs['default_backends']
    default_backends.append('triton_shared')
