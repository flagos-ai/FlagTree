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
"""Explicit asynchronous global-to-shared copies for MetaX."""

import triton.language.core as tl

from .. import types as tle
from ..semantic import get_memdesc


@tl.builtin
def async_copy_global_to_local(src, dst, mask=None, _semantic=None):
    """Issue a global-pointer-to-shared-buffer copy; the caller handles synchronization.

    Shapes and element types must match. Mask must be uniform within each
    vectorized copy; masked copies are filled with zero.
    """
    builder = _semantic.builder
    if not hasattr(builder, "create_metax_async_copy_global_to_local"):
        raise ValueError("async_copy_global_to_local is supported only on MetaX")
    if (not isinstance(src, tl.tensor) or not src.type.is_block() or not src.dtype.is_ptr()
            or not isinstance(dst, tle.buffered_tensor) or dst.type.storage != tle.smem):
        raise ValueError("MetaX asynchronous copy requires a global pointer tensor and a shared-memory destination")
    if src.dtype.address_space != 1:
        raise ValueError("MetaX asynchronous copy requires global-memory pointers")
    shape = tuple(dst.type.shape)
    if tuple(src.shape) != shape:
        raise ValueError("MetaX asynchronous copy requires pointer tensor shape and buffer shape to match")
    if src.dtype.element_ty != dst.dtype:
        raise ValueError("MetaX asynchronous copy requires pointers with the buffer element type")

    mask = tl._unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
        if mask.dtype != tl.int1:
            raise ValueError("MetaX asynchronous copy mask must have boolean element type")
        src, mask = _semantic.broadcast_impl_value(src, mask)
        if tuple(src.shape) != shape:
            raise ValueError("MetaX asynchronous copy mask must broadcast to the copy shape")

    if mask is None:
        mask = tl.full(shape, True, tl.int1, _semantic=_semantic)
    other = tl.full(shape, 0, dst.dtype, _semantic=_semantic)
    builder.create_metax_async_copy_global_to_local(src.handle, get_memdesc(dst, _semantic), mask.handle, other.handle)
