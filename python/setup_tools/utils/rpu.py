import os
import shutil
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _link_rpu_libtriton_into_main_package():
    root = _repo_root()
    src = root / "third_party" / "rpu" / "python" / "triton" / "_C" / "libtriton.so"
    dst = root / "python" / "triton" / "_C" / "libtriton.so"
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    rel_src = os.path.relpath(src, dst.parent)
    try:
        os.symlink(rel_src, dst)
    except OSError:
        shutil.copy2(src, dst)


def skip_package_dir(package):
    return True


def get_resources_url(resource_name):
    return None


def get_resources_hash(resource_name):
    return None


def install_extension(*args, **kargs):
    _link_rpu_libtriton_into_main_package()


def post_install():
    _link_rpu_libtriton_into_main_package()
