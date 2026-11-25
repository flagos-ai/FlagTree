# Triton 3.6 XPU TLE - DSA (backend-specific) buffer allocation (`tle.dsa`)
#
# Mirrors the `tle.dsa` surface the Ascend backend exposes, mapped onto the
# TritonSDNN memory hierarchy. See `spaces.py` for the address spaces.
#
# flagtree also keeps a `tle.dsa.thrive` namespace here -- a SHMEM-shaped RMA
# API named after another backend's vendor. It is deliberately NOT provided:
# the name is ambiguous on a KL5 part, and every primitive it wrapped is
# already reachable as `tle.remote` (putmem/getmem), `tle.signal` (notify) and
# `tle.signal_wait` (wait), with `fence`/`sync` being no-ops here anyway. A
# kernel copied from flagtree that calls it fails on the missing attribute,
# which is the intended outcome.
from .math import exp, exp2, log, log2, rsqrt
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
    "exp",
    "exp2",
    "log",
    "log2",
    "rsqrt",
    "address_space",
    "buffer",
    "buffer_type",
]
