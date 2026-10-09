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
"""MetaX shared-memory loads and register permutation."""

import triton.language.core as tl

from .. import types as tle
from ..semantic import get_memdesc


@tl.builtin
def local_load(buffer, dtype=None, intrinsic=False, is_constant_offs=False, mma_mode=-1, _semantic=None):
    """Load shared memory into registers; mma_mode=2 returns packed int32 for bsm_perm."""
    if not isinstance(buffer, tle.buffered_tensor) or buffer.type.storage != tle.smem:
        raise ValueError("MetaX local_load requires a shared-memory buffer")
    dtype = tl._unwrap_if_constexpr(dtype)
    dtype = buffer.dtype if dtype is None else dtype
    intrinsic = tl._unwrap_if_constexpr(intrinsic)
    is_constant_offs = tl._unwrap_if_constexpr(is_constant_offs)
    mma_mode = tl._unwrap_if_constexpr(mma_mode)
    if not isinstance(intrinsic, bool) or not isinstance(is_constant_offs, bool):
        raise ValueError("local_load intrinsic and is_constant_offs must be compile-time booleans")
    if isinstance(mma_mode, bool) or not isinstance(mma_mode, int):
        raise ValueError("local_load mma_mode must be a compile-time integer")
    if mma_mode == 2:
        if buffer.dtype not in (tl.float16, tl.bfloat16) or dtype != tl.int32 or len(buffer.shape) != 2:
            raise ValueError("local_load mma_mode=2 requires a rank-2 float16/bfloat16 buffer and dtype=tl.int32")
    elif dtype != buffer.dtype:
        raise ValueError("local_load dtype must match the buffer element type unless mma_mode=2")
    result_type = tl.block_type(dtype, buffer.type.shape)
    handle = _semantic.builder.create_metax_local_load(result_type.to_ir(_semantic.builder),
                                                       get_memdesc(buffer, _semantic), intrinsic, is_constant_offs,
                                                       mma_mode)
    return tl.tensor(handle, result_type)


@tl.builtin
def bsm_perm(value, dtype, _semantic=None):
    """Reorder a packed MetaX B operand from local_load(mma_mode=2) for MMA."""
    dtype = tl._unwrap_if_constexpr(dtype)
    if (not isinstance(value, tl.tensor) or not value.type.is_block() or len(value.shape) != 2
            or value.dtype != tl.int32 or dtype not in (tl.float16, tl.bfloat16)):
        raise ValueError("bsm_perm requires a rank-2 int32 tensor and a float16/bfloat16 result dtype")
    result_type = tl.block_type(dtype, value.type.shape)
    handle = _semantic.builder.create_metax_bsm_perm(result_type.to_ir(_semantic.builder), value.handle)
    return tl.tensor(handle, result_type)
