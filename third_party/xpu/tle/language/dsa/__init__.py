# Triton 3.6 XPU TLE - DSA (backend-specific) buffer allocation (`tle.dsa`)
#
# Mirrors the `tle.dsa` surface the Ascend backend exposes, mapped onto the
# TritonSDNN memory hierarchy. See `spaces.py` for the address spaces.
from .semantic import alloc, copy, fill, subview, to_buffer, to_tensor
from .spaces import L1D, L1W, TM, UNI_SRAM
from .types import address_space, buffer, buffer_type

__all__ = [
    "UNI_SRAM",
    "L1D",
    "L1W",
    "TM",
    "alloc",
    "copy",
    "fill",
    "subview",
    "to_buffer",
    "to_tensor",
    "address_space",
    "buffer",
    "buffer_type",
]
