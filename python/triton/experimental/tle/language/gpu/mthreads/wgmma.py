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

import triton.language as tl

from .. import types as tle


def _unwrap(value):
    return value.value if isinstance(value, tl.constexpr) else value


def use_auto_shared_layout(layout, nv_mma_shared_layout) -> bool:
    """Whether an implicit mthreads shared layout may be selected by SQMMA."""
    return layout is None and bool(_unwrap(nv_mma_shared_layout))


def mark_auto_shared_layout(builder, handle) -> None:
    if not hasattr(builder, "mark_musa_tle_auto_shared_layout"):
        raise RuntimeError("mthreads TLE native binding mark_musa_tle_auto_shared_layout is unavailable")
    builder.mark_musa_tle_auto_shared_layout(handle)


def validate_operands(a, b, acc) -> None:
    for name, operand in (("a", a), ("b", b)):
        operand_type = getattr(operand, "type", None)
        if not hasattr(operand_type, "storage"):
            raise ValueError(f"initial mthreads TLE wgmma requires shared-memory memdesc {name}")
        shape = tuple(int(_unwrap(dim)) for dim in operand_type.shape)
        if len(shape) != 2:
            raise ValueError(f"initial mthreads TLE wgmma requires rank-2 {name}, got {shape}")
        if operand.dtype not in (tl.float16, tl.bfloat16, tl.float8e4nv):
            raise ValueError("mthreads TLE wgmma requires f16, bf16, or fp8e4nv "
                             f"{name}, got {operand.dtype}")
    if a.dtype != b.dtype:
        raise ValueError("mthreads TLE wgmma requires matching A/B dtypes, "
                         f"got {a.dtype} and {b.dtype}")
    if acc is not None and getattr(acc, "dtype", None) != tl.float32:
        raise ValueError("mthreads TLE wgmma requires an f32 accumulator")


def _transpose_smem_operand(operand, semantic):
    order = [1, 0]
    handle = semantic.builder.create_memdesc_trans(operand.handle, order)
    shape = [operand.type.shape[index] for index in order]

    alloc_shape = operand.type.alloc_shape
    leading_rank = len(alloc_shape) - len(operand.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[index] for index in order]

    layout = operand.type.layout.make_permute(order)
    return tle.buffered_tensor(
        handle,
        operand.dtype,
        shape,
        operand.type.storage,
        layout,
        semantic,
        alloc_shape=transposed_alloc_shape,
    )


def prepare_operands(a, b, acc, trans_a: bool, trans_b: bool, semantic):
    """Validate mthreads SQMMA operands and build descriptor transpose views."""
    validate_operands(a, b, acc)
    if trans_a:
        a = _transpose_smem_operand(a, semantic)
    if trans_b:
        b = _transpose_smem_operand(b, semantic)
    return a, b


def validate_options(max_num_imprecise_acc: int, out_dtype) -> None:
    if max_num_imprecise_acc != 0:
        raise ValueError("mthreads TLE wgmma requires max_num_imprecise_acc=0")
    if out_dtype != tl.float32:
        raise ValueError("mthreads TLE wgmma requires out_dtype=tl.float32")


def validate_dimensions(m: int, n: int) -> None:
    if m < 16 or m % 16 != 0:
        raise ValueError("mthreads TLE wgmma result M dimension must be divisible by 16")
    if n < 16 or n % 16 != 0:
        raise ValueError("mthreads TLE wgmma result N dimension must be divisible by 16")


def validate_wait_pendings(pendings: int) -> None:
    if pendings != 0:
        raise ValueError("mthreads TLE wgmma_wait currently requires pendings=0; "
                         "non-zero pending groups are not supported")
