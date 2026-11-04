"""Spacemit backend build and dependency-cache hooks."""

import os
from pathlib import Path


def _get_backend_root() -> str:
    """Return the backend's third_party submodule path."""
    return str(Path(__file__).resolve().parents[3] / "third_party" / "spacemit")


def _collect_spacemit_package_data(backend):
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
        ("SPINE_MLIR", "spine_mlir_installed", "SPINE_MLIR_INSTALL_DIR", _set_env_path("SPINE_MLIR_INSTALL_DIR")),
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
    import glob
    import shutil
    from collections import defaultdict
    from python.build_helpers import get_cmake_dir
    from python.setup_tools.utils.tools import flagtree_configs

    cmake_dir = get_cmake_dir()
    backend_name = flagtree_configs.flagtree_backend
    backend_src_dir = Path(__file__).resolve().parents[3] / "third_party" / backend_name

    backend_dir = os.path.join(backend_src_dir, "backend")
    bin_dir = os.path.join(backend_dir, "bin")
    print(f"[spacemit] Preparing bin_dir: {bin_dir}")
    if os.path.isdir(bin_dir):
        spine_mlir_install_dir = os.environ.get("SPINE_MLIR_INSTALL_DIR")
        if spine_mlir_install_dir:
            for bin_file in ("llc", "spine-opt", "mlir-translate", "opt"):
                src = os.path.join(spine_mlir_install_dir, "bin", bin_file)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(bin_dir, bin_file))
                    print(f"[spacemit] Copied {bin_file} to backend/bin/")

        # spine-triton-opt is built by this backend rather than shipped in
        # SPINE_MLIR_INSTALL_DIR. Copy it when reusing an existing build tree.
        spine_triton_opt_src = os.path.join(str(cmake_dir), "third_party", "spacemit", "tools", "spine-triton-opt",
                                            "spine-triton-opt")
        if os.path.exists(spine_triton_opt_src):
            dst = os.path.join(bin_dir, "spine-triton-opt")
            shutil.copy(spine_triton_opt_src, dst)
            os.chmod(dst, 0o755)
            print("[spacemit] Copied spine-triton-opt to backend/bin/")

    lib_dir = os.path.join(backend_dir, "lib")
    if os.path.isdir(lib_dir):
        so_file_list = []
        spine_mlir_install_dir = os.environ.get("SPINE_MLIR_INSTALL_DIR")
        if spine_mlir_install_dir:
            for pattern in ("libSpeIR*.so*", "libspine_tcm.so*"):
                so_file_list.extend(glob.glob(os.path.join(spine_mlir_install_dir, "lib", pattern)))

        spine_runtime_install_dir = os.environ.get("SPINE_RUNTIME_INSTALL_DIR")
        if spine_runtime_install_dir:
            so_file_list.extend(glob.glob(os.path.join(spine_runtime_install_dir, "lib", "libspert.so*")))

        lib_groups = defaultdict(list)
        for so_file in so_file_list:
            basename = os.path.basename(so_file)
            lib_groups[basename.split(".so")[0]].append(so_file)
        for files in lib_groups.values():
            real_file = max(files, key=lambda f: len(os.path.basename(f)))
            real_base = os.path.basename(real_file)
            shutil.copy(os.path.realpath(real_file), os.path.join(lib_dir, real_base))
            print(f"[spacemit] Copied {real_base} to backend/lib/")
            # Recreate the SONAME symlink chain so transitive dependencies
            # such as libspert.so.0 resolve to the versioned real file.
            for so_file in files:
                link_base = os.path.basename(so_file)
                if link_base == real_base:
                    continue
                link_path = os.path.join(lib_dir, link_base)
                if os.path.islink(link_path) or os.path.exists(link_path):
                    os.unlink(link_path)
                os.symlink(real_base, link_path)
                print(f"[spacemit] Symlink {link_base} -> {real_base}")

    spine_runtime_install_dir = os.environ.get("SPINE_RUNTIME_INSTALL_DIR")
    if spine_runtime_install_dir:
        include_dir = os.path.join(backend_dir, "include", "SpineRuntime")
        os.makedirs(include_dir, exist_ok=True)
        for header in ("spert.hpp", "spert_engine.hpp", "spert_abi.h"):
            src = os.path.join(spine_runtime_install_dir, "include", header)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(include_dir, header))
                print(f"[spacemit] Vendored {header} from spine-runtime")

    # Vendor the MLIR Python bindings (mlir_core) into the backend so the
    # wheel is self-contained: llvm_direct.py / mixed_bridge.py (the smt_rvisa
    # LLVM-direct emitter) import them at module load, and _mlir_loader.py
    # resolves backend/mlir_core at runtime with no PYTHONPATH.
    # Source resolution: an LLVM install with python_packages/mlir_core, a
    # "-python" release package (a directory containing mlir_core/), or the
    # mlir_core directory itself — via SPINE_MLIR_BINDINGS_DIR or $LLVM_SYSPATH.
    mlir_bindings_root = os.environ.get("SPINE_MLIR_BINDINGS_DIR") or os.environ.get("LLVM_SYSPATH")
    if mlir_bindings_root:
        mlir_bindings_src = None
        for sub in (os.path.join("python_packages", "mlir_core"), "mlir_core", ""):
            cand = os.path.join(mlir_bindings_root, sub) if sub else mlir_bindings_root
            if os.path.isdir(os.path.join(cand, "mlir")):
                mlir_bindings_src = cand
                break
        if mlir_bindings_src is None:
            raise RuntimeError("[spacemit] MLIR Python bindings (mlir_core) not found under "
                               f"{mlir_bindings_root} — pass the -python release package or an "
                               "LLVM install with python_packages/mlir_core via "
                               "SPINE_MLIR_BINDINGS_DIR / LLVM_SYSPATH (smt_rvisa llvm_direct "
                               "needs them).")
        mlir_bindings_dst = os.path.join(backend_dir, "mlir_core")
        if os.path.isdir(mlir_bindings_dst):
            shutil.rmtree(mlir_bindings_dst)
        shutil.copytree(mlir_bindings_src, mlir_bindings_dst)
        print(f"[spacemit] Vendored MLIR Python bindings to backend/mlir_core/ "
              f"(from {mlir_bindings_src})")
