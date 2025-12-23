def precompile_hock(*args, **kargs):
    default_backends = kargs['default_backends']
    default_backends.append('triton_shared')


def get_resources_url(resource_name):
    ...


def get_resources_hash(resource_name):
    ...
