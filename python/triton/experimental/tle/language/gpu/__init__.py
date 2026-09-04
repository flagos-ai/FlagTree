# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# flagtree tle
from .core import (
    pipeline,
    range,
    alloc,
    alloc_barrier,
    alloc_barriers,
    barrier_arrive,
    barrier_wait,
    copy,
    memory_space,
    local_ptr,
    warp_specialize,
    wgmma,
    wgmma_wait,
    set_layout,
)
from .types import (layout, distributed_encoding, BlockEncoding, MmaEncoding, DotOperandEncoding, SlicedEncoding,
                    shared_layout, swizzled_shared_layout, tensor_memory_layout, nv_mma_shared_layout, scope,
                    buffered_tensor, buffered_tensor_type, barrier, barrier_type, smem, tmem, PENDING, READY)

# Backward-compat alias expected by existing tests/tutorials.
storage_kind = memory_space

__all__ = [
    "pipeline",
    "range",
    "alloc",
    "alloc_barrier",
    "alloc_barriers",
    "barrier_arrive",
    "barrier_wait",
    "copy",
    "local_ptr",
    "warp_specialize",
    "wgmma",
    "wgmma_wait",
    "set_layout",
    "storage_kind",
    "layout",
    "distributed_encoding",
    "BlockEncoding",
    "MmaEncoding",
    "DotOperandEncoding",
    "SlicedEncoding",
    "memory_space",
    "shared_layout",
    "swizzled_shared_layout",
    "tensor_memory_layout",
    "nv_mma_shared_layout",
    "scope",
    "buffered_tensor",
    "buffered_tensor_type",
    "barrier",
    "barrier_type",
    "PENDING",
    "READY",
    "smem",
    "tmem",
]
