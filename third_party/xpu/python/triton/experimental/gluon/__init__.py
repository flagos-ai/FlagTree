# flagtree backend path specialization
from triton.flagtree_spec import spec_path

spec_path(__path__)

from triton._flagtree_backend import FLAGTREE_BACKEND
from . import nvidia
# flagtree backend path specialization
if FLAGTREE_BACKEND == "hcu":
    from . import hcu
else:
    from . import amd
from ._runtime import constexpr_function, jit
from triton.language.core import must_use_result

__all__ = ["constexpr_function", "jit", "must_use_result", "nvidia", "hcu" if FLAGTREE_BACKEND == "hcu" else "amd"]
