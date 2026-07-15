from triton.flagtree_spec import spec_path

# flagtree backend path specialization
spec_path(__path__)

from . import blackwell
from . import hopper

__all__ = ["blackwell", "hopper"]
