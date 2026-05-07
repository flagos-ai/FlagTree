import shutil
from pathlib import Path


def install_extension(*args, **kargs):
    """Copy third_party/hcu/python/triton to python/triton """
    _python_dir = Path(__file__).parent.parent.parent
    src_dir = f"{_python_dir}/../third_party/hcu/python/triton"
    dst_dir = f"{_python_dir}/triton"
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
