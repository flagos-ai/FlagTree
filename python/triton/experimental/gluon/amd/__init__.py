from triton.flagtree_spec import spec_path

# flagtree backend path specialization
spec_path(__path__)

from . import gfx1250

__all__ = ["gfx1250"]
