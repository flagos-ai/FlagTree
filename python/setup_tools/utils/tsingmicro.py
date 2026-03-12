import os


def precompile_hock(*args, **kargs):
    default_backends = kargs["default_backends"]
    default_backends.append('flir')


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return [
        "-DCMAKE_INSTALL_PREFIX=" + src_ext_path,
    ]
