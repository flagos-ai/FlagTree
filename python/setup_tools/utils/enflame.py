import os
import shutil
import sys
from pathlib import Path


def get_package_data_tools():
    """Declare tool files to be packaged"""
    return ["triton-gcu300-opt", "triton-gcu400-opt"]


def install_extension(*args, **kargs):
    """Copy triton-gcu400-opt to third_party/enflame/backend directory"""
    # Lazy import: build_helpers lives in python/, so add it to path first
    _python_dir = Path(__file__).parent.parent.parent
    if str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))
    from build_helpers import get_cmake_dir
    # Get CMake build directory using the same function as setup.py
    # This returns build/cmake.linux-x86_64-cpython-3.10, not build/temp.*
    cmake_dir = get_cmake_dir()
    binary_dir = cmake_dir / "bin"

    # Get project root directory (from cmake_dir go up 2 levels: build/cmake.xxx -> build -> root)
    project_root_dir = cmake_dir.parent.parent

    # Modify nvidia driver's is_active() to return False for enflame backend
    # This prevents nvidia driver from being activated when using enflame
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

    # Target directory: third_party/enflame/backend/
    # This is where the backend source files are located, and setup.py will create
    # a symlink from python/triton/backends/enflame to this directory
    dst_dir = project_root_dir / "third_party" / "enflame" / "backend"

    # Ensure target directory exists
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy triton-gcu400-opt if it exists
    for target in ["triton-gcu400-opt"]:
        src_path = binary_dir / target
        dst_path = dst_dir / target

        if src_path.exists():
            print(f"Copying {src_path} -> {dst_path}")
            shutil.copy(src_path, dst_path)
            # Set executable permissions
            os.chmod(dst_path, 0o755)
        else:
            print(f"Warning: {src_path} not found, skipping")

    # Also copy triton-gcu300-opt if it exists
    for target in ["triton-gcu300-opt"]:
        src_path = binary_dir / target
        dst_path = dst_dir / target

        if src_path.exists():
            print(f"Copying {src_path} -> {dst_path}")
            shutil.copy(src_path, dst_path)
            os.chmod(dst_path, 0o755)
