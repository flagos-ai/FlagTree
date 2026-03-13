import os


def precompile_hock(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends_list = list(default_backends)
    default_backends_list.append('flir')
    kargs["default_backends"] = tuple(default_backends_list)
    default_backends = tuple(default_backends_list)
    return default_backends


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return [
        "-DCMAKE_INSTALL_PREFIX=" + src_ext_path,
    ]
