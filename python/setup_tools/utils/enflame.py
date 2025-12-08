import shutil
from pathlib import Path
from build_helpers import get_cmake_dir


def install_extension(*args, **kargs):
    cmake_dir = get_cmake_dir()
    binary_dir = cmake_dir / "bin"
    python_root_dir = Path(__file__).parent.parent.parent
    src_root_dir = python_root_dir.parent

    drvfile = src_root_dir / 'third_party' / 'nvidia' / 'backend' / 'driver.py'
    with open(drvfile, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'def is_active():' in line:
            if 'return False' not in lines[i + 1]:
                lines.insert(i + 1, '        return False\n')
            break
    with open(drvfile, 'w') as f:
        f.writelines(lines)

    dst_dir = python_root_dir / "triton" / "backends" / "enflame"
    for target in ["triton-gcu300-opt"]:
        src_path = binary_dir / target
        dst_path = dst_dir / target
        shutil.copy(src_path, dst_path)
