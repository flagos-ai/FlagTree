import os

try:
    from triton._flagtree_backend import FLAGTREE_BACKEND
except ModuleNotFoundError:
    FLAGTREE_BACKEND = os.environ.get("FLAGTREE_BACKEND", "")


def _has_iluvatar_libtriton() -> bool:
    try:
        from triton._C import libtriton
    except ImportError:
        return False
    return hasattr(libtriton, "iluvatar")


def enabled() -> bool:
    return FLAGTREE_BACKEND == "iluvatar" or _has_iluvatar_libtriton()
