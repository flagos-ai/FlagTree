import glob
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
