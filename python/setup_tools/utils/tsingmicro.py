import os


def get_backend_cmake_args(*args, **kargs):
    build_ext = kargs['build_ext']
    src_ext_path = build_ext.get_ext_fullpath("triton")
    src_ext_path = os.path.abspath(os.path.dirname(src_ext_path))
    return [
        "-DCMAKE_INSTALL_PREFIX=" + src_ext_path,
    ]


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    cache.store(
        file="tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64",
        condition=("tsingmicro" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm21-glibc2.30-glibcxx3.4.28-python3.11-x64_v0.2.0.tar.gz",
        pre_hook=lambda: check_env('LLVM_SYSPATH'),
        post_hook=set_llvm_env,
    )

    cache.store(
        file="tx8_deps",
        condition=("tsingmicro" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_release_20250814_195126_v0.2.0.tar.gz",
        pre_hook=lambda: check_env('TX8_DEPS_ROOT'),
        post_hook=lambda path: os.environ.update({'LLVM_SYSPATH': path}),
    )


def get_package_data(backends):
    """Declare non-Python files the tsingmicro backend needs at runtime."""
    from pathlib import Path

    package_data = {}
    for backend in backends:
        if backend.name != "tsingmicro":
            continue
        backend_dir = Path(backend.backend_dir)
        files = []
        for path in backend_dir.rglob("*"):
            if not path.is_file() or path.suffix in (".py", ".pyc", ".pyo"):
                continue
            files.append(str(path.relative_to(backend_dir)))
        if files:
            package_data["triton.backends.tsingmicro"] = files
        break
    return package_data
