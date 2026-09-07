import os
from pathlib import Path


def _get_backend_root() -> str:
    """Return the backend's third_party submodule path."""
    return str(Path(__file__).resolve().parents[3] / "third_party" / "tsingmicro")


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    def set_env(env_dict: dict):
        for env_k, env_v in env_dict.items():
            os.environ[env_k] = str(env_v)

    cache.store(
        file="tsingmicro-llvm22",
        condition=("tsingmicro" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tsingmicro-llvm22-x64_v0.6.1.tar.gz",
        pre_hook=lambda: check_env('LLVM_SYSPATH'),
        post_hook=set_llvm_env,
    )

    cache.store(
        file="tx8_deps",
        condition=("tsingmicro" == flagtree_backend),
        url="https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/tx8_depends_v0.6.1.tar.gz",
        pre_hook=lambda: check_env('TX8_DEPS_ROOT'),
        post_hook=lambda path: set_env({
            'TX8_DEPS_ROOT': path,
            'TX8_YOC_RT_THREAD_SMP': Path(path) / "tx8-yoc-rt-thread-smp",
        }),
    )


def _collect_tsingmicro_package_data(backend):
    """Collect non-Python files the backend needs at runtime."""
    backend_dir = Path(backend.backend_dir)
    files = []
    if not backend_dir.is_dir():
        return files
    for path in backend_dir.rglob("*"):
        if not path.is_file() or path.suffix in (".py", ".pyc", ".pyo"):
            continue
        files.append(str(path.relative_to(backend_dir)))
    return files


def get_package_data(backends):
    """Declare non-Python files the tsingmicro backend needs at runtime."""
    package_data = {}
    for backend in backends:
        if backend.name != "tsingmicro":
            continue
        files = _collect_tsingmicro_package_data(backend)
        if files:
            package_data["triton.backends.tsingmicro"] = files
        break
    return package_data


def overlay_runtime_so(cache, build_py_command=None, backends=None):
    """Re-scan backend files after build_ext has produced runtime artifacts.

    bin/lib artifacts (tsingmicro-opt, tx-profiler, libvr.a) are created by the
    install_extension hook during build_ext, so setup() evaluation of
    get_package_data runs before they exist. Re-scan after build_ext and push
    the collected files back into package_data for build_py.
    """
    if build_py_command is None or backends is None:
        return
    build_py_command.distribution.package_data = build_py_command.distribution.package_data or {}
    for backend in backends:
        if backend.name != "tsingmicro":
            continue
        files = _collect_tsingmicro_package_data(backend)
        if files:
            build_py_command.distribution.package_data["triton.backends.tsingmicro"] = files
        build_py_command.package_data = build_py_command.distribution.package_data
        build_py_command.__dict__.pop("data_files", None)
        break


def install_extension(*args, **kargs):
    """Copy built backend artifacts from the CMake build tree into backend/bin, backend/lib."""
    import shutil
    import sys

    python_dir = Path(__file__).resolve().parents[2]
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from build_helpers import get_cmake_dir

    build_temp = Path(get_cmake_dir())
    backend_root = Path(_get_backend_root())

    # Build-tree artifact locations (see backend bin/crt/profiler CMakeLists):
    # - tsingmicro-opt -> third_party/tsingmicro/bin/tsingmicro-opt
    # - libvr.a        -> third_party/tsingmicro/crt/lib/libvr.a
    # - tx-profiler    -> bin/tx-profiler (RUNTIME_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/bin)
    artifacts = {
        "bin": [
            build_temp / "third_party" / "tsingmicro" / "bin" / "tsingmicro-opt",
            build_temp / "bin" / "tx-profiler",
        ],
        "lib": [
            build_temp / "third_party" / "tsingmicro" / "crt" / "lib" / "libvr.a",
        ],
    }

    # Clean stale artifacts first so outdated binaries are never repackaged.
    for sub in ("bin", "lib"):
        dst_dir = backend_root / "backend" / sub
        if dst_dir.is_dir():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

    for sub, src_paths in artifacts.items():
        dst_dir = backend_root / "backend" / sub
        for src_path in src_paths:
            if not src_path.exists():
                print(f"[tsingmicro] Warning: {src_path} not found, skipping")
                continue
            dst_path = dst_dir / src_path.name
            shutil.copy(src_path, dst_path)
            dst_path.chmod(0o755)
            print(f"[tsingmicro] Copied {src_path} to {dst_path}")
