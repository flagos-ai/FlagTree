from triton.flagtree_spec import spec_path

# flagtree backend path specialization
spec_path(__path__)

from ._runtime import constexpr_function, jit
from triton.language.core import must_use_result
from . import nvidia
from triton._flagtree_backend import FLAGTREE_BACKEND
if FLAGTREE_BACKEND == "hcu":  # flagtree hcu
    from . import hcu
else:
    from . import amd

__all__ = ["constexpr_function", "jit", "must_use_result", "nvidia", "hcu" if FLAGTREE_BACKEND == "hcu" else "amd"]
