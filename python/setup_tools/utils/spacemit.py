"""Spacemit backend build and dependency-cache hooks."""

import os


def _get_backend_root() -> str:
    """Return the backend's third_party submodule path."""
    from pathlib import Path
    return str(Path(__file__).resolve().parents[3] / "third_party" / "spacemit")


def _collect_spacemit_package_data(backend):
    """Collect non-Python files the backend needs at runtime."""
    from pathlib import Path
    backend_dir = Path(backend.backend_dir)
    files = []
    if not backend_dir.is_dir():
        return files
    for path in backend_dir.rglob("*"):
        if not path.is_file() or path.suffix in (".py", ".pyc", ".pyo"):
            continue
        files.append(str(path.relative_to(backend_dir)))
    return files


def _cache_asset(name):
    """Read one CI asset URL and version from the sourced environment."""
    url = os.environ.get(f"{name}_URL")
    version = os.environ.get(f"{name}_MD5")
    if not url and not version:
        return None
    if not url or not version:
        raise RuntimeError(f"{name}_URL and {name}_MD5 must be set together")
    return url, version


def _set_env_path(name):
    def set_path(path):
        os.environ[name] = str(path)

    return set_path


def register_cache(cache, flagtree_backend, check_env, set_llvm_env):
    """Register Spacemit build, cross-toolchain, and QEMU dependencies."""
    if flagtree_backend != "spacemit":
        return

    assets = (
        ("LLVM", "llvm_installed", "LLVM_SYSPATH", set_llvm_env),
        ("SPINE_MLIR", "spine_mlir_installed", "SPINE_MLIR_INSTALL_DIR",
         _set_env_path("SPINE_MLIR_INSTALL_DIR")),
        ("SPINE_RUNTIME", "spine_runtime_installed", "SPINE_RUNTIME_INSTALL_DIR",
         _set_env_path("SPINE_RUNTIME_INSTALL_DIR")),
        ("RPC_RUNTIME", "rpc_runtime_installed", None, None),
        ("RISCV_TOOLCHAIN", "toolchain", None, None),
        ("JDSK_QEMU", "jdsk-qemu", None, None),
    )
    for env_prefix, cache_name, override_env, post_hook in assets:
        asset = _cache_asset(env_prefix)
        if asset is None:
            continue
        url, version = asset
        cache.store(
            file=cache_name,
            condition=True,
            url=url,
            version=version,
            pre_hook=(lambda name=override_env: check_env(name)) if override_env else None,
            post_hook=post_hook,
        )


def get_package_data(backends):
    package_data = {}
    for backend in backends:
        if backend.name == "spacemit":
            files = _collect_spacemit_package_data(backend)
            if files:
                package_data["triton.backends.spacemit"] = files
            break
    return package_data


def overlay_runtime_so(cache, build_py_command=None, backends=None):
    """Re-scan backend files after build_ext has produced runtime artifacts."""
    if build_py_command is None or backends is None:
        return
    build_py_command.distribution.package_data = build_py_command.distribution.package_data or {}
    for backend in backends:
        if backend.name == "spacemit":
            files = _collect_spacemit_package_data(backend)
            if files:
                build_py_command.distribution.package_data["triton.backends.spacemit"] = files
            build_py_command.package_data = build_py_command.distribution.package_data
            build_py_command.__dict__.pop("data_files", None)
            break


def get_backend_cmake_args(*args, **kargs):
    """Inject LLVM/MLIR/LLD cmake args, and create build-tree symlinks."""
    cmake_args = []

    llvm_syspath = os.environ.get("LLVM_SYSPATH")
    if llvm_syspath:
        cmake_args += [
            f"-DLLVM_DIR={llvm_syspath}/lib/cmake/llvm",
            f"-DMLIR_DIR={llvm_syspath}/lib/cmake/mlir",
            f"-DLLD_DIR={llvm_syspath}/lib/cmake/lld",
            f"-DLLVM_LIBRARY_DIR={llvm_syspath}/lib",
        ]

    # CpuProfiler.cpp is not available for this backend.
    cmake_args.append("-DTRITON_BUILD_PROTON=OFF")

    build_ext = kargs.get("build_ext")
    if build_ext and hasattr(build_ext, "build_temp") and build_ext.build_temp:
        build_third_party = os.path.join(build_ext.build_temp, "third_party")
        if os.path.isdir(build_third_party):
            spine_triton_link = os.path.join(build_third_party, "spine_triton")
            if not os.path.exists(spine_triton_link):
                try:
                    os.symlink("spacemit", spine_triton_link)
                    print(f"[spacemit] Created build-tree symlink: {spine_triton_link} -> spacemit")
                except (OSError, PermissionError) as error:
                    print(f"[spacemit] Warning: could not create build-tree symlink: {error}")

    return cmake_args


def precompile_hook(*args, **kargs):
    """Pre-build setup hook."""
    return None


def post_install():
    """Create build-tree symlinks needed for generated include files."""
    try:
        from pathlib import Path
        backend_root = _get_backend_root()
        build_dirs = list(Path(backend_root, "triton", "build").glob("cmake.*"))
        for build_dir in build_dirs:
            build_third_party = build_dir / "third_party"
            if build_third_party.is_dir():
                spine_triton_link = build_third_party / "spine_triton"
                if not spine_triton_link.exists():
                    spine_triton_link.symlink_to("spacemit")
                    print(f"[spacemit] Created build-tree symlink: {spine_triton_link} -> spacemit")
    except Exception:
        pass


def install_extension(*args, **kargs):
    """Copy spine-triton-opt from the CMake build directory."""
    import shutil
    import sys
    from pathlib import Path

    python_dir = Path(__file__).resolve().parents[2]
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    from build_helpers import get_cmake_dir

    build_temp = Path(get_cmake_dir())
    spine_triton_opt_src = (
        build_temp / "third_party" / "spacemit" / "tools" /
        "spine-triton-opt" / "spine-triton-opt"
    )
    if not spine_triton_opt_src.exists():
        print(f"[spacemit] Warning: spine-triton-opt not found at {spine_triton_opt_src}")
        return

    backend_bin = Path(_get_backend_root()) / "backend" / "bin"
    backend_bin.mkdir(parents=True, exist_ok=True)
    spine_triton_opt_dst = backend_bin / "spine-triton-opt"
    shutil.copy(spine_triton_opt_src, spine_triton_opt_dst)
    spine_triton_opt_dst.chmod(0o755)
    print(f"[spacemit] Copied spine-triton-opt to {spine_triton_opt_dst}")
