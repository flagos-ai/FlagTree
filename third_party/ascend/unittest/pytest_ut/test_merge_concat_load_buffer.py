# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""End-to-end coverage for the merge-concat-load-buffer pass.

Two masked tl.load results that are concatenated lower to two full-tile UB
allocs, two linalg.fill ops and an extract_slice/insert_slice pair. The pass
folds the two allocs into one and drops the fills when it can prove the copies
cover every later read.

The kernels below reproduce that shape so the optimised code is exercised on
device. Padding correctness is the interesting part: if the pass ever dropped a
fill it could not prove dead, the `other` value would be replaced by whatever
was left in UB, so `test_cat_masked_load_padding_visible` stores the whole tile
instead of masking the tail away.

The pipeline only enables the pass on 910_95/950, so on other targets these
tests cover the unoptimised lowering of the same kernels rather than the pass.
"""

import pytest
import torch
import torch_npu

import triton
import triton.language as tl

import test_common


@triton.jit
def fused_cat_masked_load_kernel(in_ptr0, in_ptr1, out_ptr, y_numel, HALF: tl.constexpr, YBLOCK: tl.constexpr,
                                 OTHER: tl.constexpr, MASK_STORE: tl.constexpr):
    """Concatenate two row-masked loads along the column axis.

    This mirrors what torch-inductor emits for a `cat` of two tensors: one
    full-width tile whose left half is loaded from `in_ptr0` and whose right
    half is loaded from `in_ptr1` at a negative bias.
    """
    yoffset = tl.program_id(0) * YBLOCK
    row = (yoffset + tl.arange(0, YBLOCK))[:, None]
    col = tl.arange(0, 2 * HALF)[None, :]

    row_mask = row < y_numel
    from_left = col < HALF

    left = tl.load(in_ptr0 + row * HALF + col, mask=row_mask & from_left, other=OTHER)
    right = tl.load(in_ptr1 + row * HALF + (col - HALF), mask=row_mask & (col >= HALF), other=OTHER)
    result = tl.where(from_left, left, right)

    if MASK_STORE:
        tl.store(out_ptr + row * (2 * HALF) + col, result, mask=row_mask)
    else:
        tl.store(out_ptr + row * (2 * HALF) + col, result)


@triton.jit
def cat_masked_load_1d_kernel(x_ptr, y_ptr, out_ptr, n_valid, BLOCK: tl.constexpr, OTHER: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    mask = idx < n_valid

    x = tl.load(x_ptr + idx, mask=mask, other=OTHER)
    y = tl.load(y_ptr + idx, mask=mask, other=OTHER)
    result = tl.cat(x, y, can_reorder=True)

    tl.store(out_ptr + tl.arange(0, 2 * BLOCK), result)


def _reference(x0, x1, rows, half, y_valid, tail):
    """Expected output tile: valid rows carry the concat, the rest carry `tail`."""
    ref = torch.full((rows, 2 * half), tail, dtype=x0.dtype)
    ref[:y_valid, :half] = x0[:y_valid]
    ref[:y_valid, half:] = x1[:y_valid]
    return ref


# y_numel is deliberately not a multiple of YBLOCK in most cases so the tail
# tile is partially masked and the padding path is actually taken.
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'int32'])
@pytest.mark.parametrize('y_numel, yblock, half', [
    (32, 32, 256),
    (30, 32, 256),
    (70, 32, 128),
    (1, 32, 64),
    (100, 16, 32),
])
def test_cat_masked_load_masked_store(dtype, y_numel, yblock, half):
    """Concat of two masked loads, tail rows masked away on store."""
    torch_dtype = eval('torch.' + dtype)
    n_prog = triton.cdiv(y_numel, yblock)
    rows = n_prog * yblock
    sentinel = -1

    x0 = test_common.generate_tensor((rows, half), dtype)
    x1 = test_common.generate_tensor((rows, half), dtype)
    out = torch.full((rows, 2 * half), sentinel, dtype=torch_dtype)

    # Tail rows are never stored, so they keep the sentinel the buffer held.
    ref = _reference(x0, x1, rows, half, y_valid=y_numel, tail=sentinel)

    out_npu = out.npu()
    fused_cat_masked_load_kernel[n_prog, 1, 1](x0.npu(), x1.npu(), out_npu, y_numel, HALF=half, YBLOCK=yblock, OTHER=0,
                                               MASK_STORE=True)

    test_common.validate_cmp(dtype, out_npu, ref)


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('y_numel, yblock, half', [
    (30, 32, 256),
    (70, 32, 128),
    (1, 32, 64),
])
def test_cat_masked_load_padding_visible(dtype, y_numel, yblock, half):
    """Same concat, but the whole tile is stored so `other` padding is observed.

    This is the case where the pass must not drop the fill it cannot prove
    dead: rows in [y_numel, rows) are never written by a copy, so their value
    comes purely from the padding.
    """
    torch_dtype = eval('torch.' + dtype)
    n_prog = triton.cdiv(y_numel, yblock)
    rows = n_prog * yblock
    other = -7.5

    x0 = test_common.generate_tensor((rows, half), dtype)
    x1 = test_common.generate_tensor((rows, half), dtype)
    out = torch.zeros((rows, 2 * half), dtype=torch_dtype)

    ref = _reference(x0, x1, rows, half, y_valid=y_numel, tail=other)

    out_npu = out.npu()
    fused_cat_masked_load_kernel[n_prog, 1, 1](x0.npu(), x1.npu(), out_npu, y_numel, HALF=half, YBLOCK=yblock,
                                               OTHER=other, MASK_STORE=False)

    test_common.validate_cmp(dtype, out_npu, ref)


@pytest.mark.parametrize('dtype', ['float32', 'float16'])
@pytest.mark.parametrize('block, n_valid', [
    (64, 64),
    (64, 40),
    (128, 1),
])
def test_cat_masked_load_1d(dtype, block, n_valid):
    """1-D tl.cat of two masked loads; the padding is always observable."""
    torch_dtype = eval('torch.' + dtype)
    other = -3.25

    x = test_common.generate_tensor((block, ), dtype)
    y = test_common.generate_tensor((block, ), dtype)

    keep = torch.arange(block) < n_valid
    pad = torch.full((block, ), other, dtype=torch_dtype)
    ref = torch.cat((torch.where(keep, x, pad), torch.where(keep, y, pad)))

    out_npu = torch.zeros((2 * block, ), dtype=torch_dtype).npu()
    cat_masked_load_1d_kernel[1, 1, 1](x.npu(), y.npu(), out_npu, n_valid, BLOCK=block, OTHER=other)

    test_common.validate_cmp(dtype, out_npu, ref)
