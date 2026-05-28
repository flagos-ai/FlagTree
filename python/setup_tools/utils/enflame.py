import glob
import importlib.util
import os
import shutil
import sys
from pathlib import Path


def get_package_data_tools():
    """Declare tool files to be packaged"""
    return [
        "triton-gcu300-opt",
        "triton-gcu400-opt",
        "libtriton_gcu300_core.so",
        "libtriton_gcu400_core.so",
        "_triton_gcu300*.so",
        "_triton_gcu400*.so",
    ]


def _find_mlir_core_path():
    """Find MLIR Python bindings (mlir_core) from LLVM env vars or flagtree cache."""
    search_roots = []
    for env_var in ("LLVM_SYSPATH", "KURAMA_LLVM_DIR"):
        val = os.environ.get(env_var, "")
        if val:
            search_roots.append(val)

    home = os.environ.get("HOME", os.path.expanduser("~"))
    flagtree_cache = os.environ.get("FLAGTREE_CACHE_DIR", os.path.join(home, ".flagtree"))
    enflame_cache = os.path.join(flagtree_cache, "enflame")
    if os.path.isdir(enflame_cache):
        for entry in os.listdir(enflame_cache):
            entry_path = os.path.join(enflame_cache, entry)
            if os.path.isdir(entry_path) and "llvm" in entry.lower():
                search_roots.append(entry_path)

    for root in search_roots:
        candidate = os.path.join(root, "python_packages", "mlir_core")
        if os.path.isdir(os.path.join(candidate, "mlir")):
            return candidate

    try:
        spec = importlib.util.find_spec("mlir")
        if spec and spec.origin:
            return None
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def install_extension(*args, **kargs):
    """Copy GCU binaries and shared libraries to the backend directory."""
    _python_dir = Path(__file__).parent.parent.parent
    if str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))
    from build_helpers import get_cmake_dir

    cmake_dir = get_cmake_dir()
    binary_dir = cmake_dir / "bin"
    lib_dir = cmake_dir / "lib"

    project_root_dir = cmake_dir.parent.parent

    # Modify nvidia driver's is_active() to return False for enflame backend
    drvfile = project_root_dir / 'third_party' / 'nvidia' / 'backend' / 'driver.py'
    if drvfile.exists():
        with open(drvfile, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'def is_active():' in line:
                if i + 1 < len(lines) and 'return False' not in lines[i + 1]:
                    lines.insert(i + 1, '        return False\n')
                break
        with open(drvfile, 'w') as f:
            f.writelines(lines)

    dst_dir = project_root_dir / "third_party" / "enflame" / "backend"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy triton-gcu*-opt executables from bin/
    for target in ["triton-gcu300-opt", "triton-gcu400-opt"]:
        src_path = binary_dir / target
        dst_path = dst_dir / target
        if src_path.exists():
            print(f"Copying {src_path} -> {dst_path}")
            shutil.copy(src_path, dst_path)
            os.chmod(dst_path, 0o755)
        else:
            print(f"Warning: {src_path} not found, skipping")

    # Copy core shared libraries and Python binding .so from lib/
    # toolkit.py expects these next to the backend directory
    so_patterns = [
        "libtriton_gcu300_core.so*",
        "libtriton_gcu400_core.so*",
        "_triton_gcu300*.so",
        "_triton_gcu400*.so",
    ]
    for pattern in so_patterns:
        for src_path in sorted(glob.glob(str(lib_dir / pattern))):
            src_path = Path(src_path)
            dst_path = dst_dir / src_path.name
            print(f"Copying {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)

    # Copy MLIR Python bindings to build_lib for packaging
    build_ext = kargs.get('build_ext')
    if build_ext and hasattr(build_ext, 'build_lib'):
        mlir_core_path = _find_mlir_core_path()
        if mlir_core_path:
            mlir_src = os.path.join(mlir_core_path, "mlir")
            mlir_dst = os.path.join(build_ext.build_lib, "mlir")
            if os.path.isdir(mlir_src):
                if os.path.exists(mlir_dst):
                    shutil.rmtree(mlir_dst)
                shutil.copytree(mlir_src, mlir_dst)
                print(f"Copied MLIR Python bindings from {mlir_src} to {mlir_dst}", file=sys.stderr)
