import shutil
from pathlib import Path

def install_extension(*args, **kargs):
    # Copy knobs.py
    _python_dir = Path(__file__).parent.parent.parent
    src_triton_dir = _python_dir / "../third_party/metax/python/triton"
    dst_triton_dir = _python_dir / "triton"
    shutil.copy(src_triton_dir / "knobs.py", dst_triton_dir / "knobs.py")

    # copy runtime/driver.py
    shutil.copy(src_triton_dir / "runtime/driver.py", dst_triton_dir / "runtime/driver.py")
    # copy language/core.py semantic.py target_info.py
    shutil.copytree(src_triton_dir / "language", dst_triton_dir / "language", dirs_exist_ok=True)
