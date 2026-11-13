# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def _mixed_axis_bounds(input_ptr, output_ptr, rows, lower, upper, TENSOR_OTHER: tl.constexpr):
    x = tl.arange(0, 3)[:, None]
    r = tl.arange(0, 32)[None, :]
    # A nonstructured row and a structured, strided column. Only the masked
    # interval has storage; a nonzero lower bound makes preceding addresses
    # invalid too. Empty intervals must perform no reads.
    ptr = input_ptr + (x % 2) + (r - lower) * 16
    mask = (x < rows) & (r >= lower) & (r < upper)
    other = (x * 100 + r).to(tl.float32) if TENSOR_OTHER else -7.0
    value = tl.load(ptr, mask, other=other)
    tl.store(output_ptr + x * 32 + r, value)


@pytest.mark.parametrize("lower,upper", [(0, 32), (0, 20), (7, 23), (20, 7), (32, 32)])
@pytest.mark.parametrize("rows", [0, 2, 3])
@pytest.mark.parametrize("tensor_other", [False, True])
def test_mixed_axis_bounds(lower, upper, rows, tensor_other):
    source = torch.arange(max(upper - lower, 1) * 16, dtype=torch.float32).reshape(-1, 16)
    x = torch.arange(3)[:, None]
    r = torch.arange(32)[None, :]
    expected = (x * 100 + r).float() if tensor_other else torch.full((3, 32), -7.0)
    for row in range(rows):
        if upper > lower:
            expected[row, lower:upper] = source[:, row % 2]
    output = torch.empty((3, 32), dtype=torch.float32, device="npu")
    _mixed_axis_bounds[(1, )](source.npu(), output, rows, lower, upper, tensor_other, multibuffer=False)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _mixed_axis_variance(input_ptr, output_ptr, xnumel, rnumel, WIDTH: tl.constexpr, XBLOCK: tl.constexpr,
                         RBLOCK: tl.constexpr):
    x = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    xmask = x < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    accum = tl.full((XBLOCK, RBLOCK), 0, tl.float32)
    for start in range(0, rnumel, RBLOCK):
        r = start + rbase
        mask = xmask & (r < rnumel)
        offset = x % WIDTH + WIDTH * r + WIDTH * rnumel * (x // WIDTH)
        value = tl.load(input_ptr + offset, mask, other=0.0)
        accum = tl.where(mask, accum + value, accum)
    mean = tl.sum(accum, 1)[:, None] / rnumel
    variance = tl.full((XBLOCK, RBLOCK), 0, tl.float32)
    for start in range(0, rnumel, RBLOCK):
        r = start + rbase
        mask = xmask & (r < rnumel)
        offset = x % WIDTH + WIDTH * r + WIDTH * rnumel * (x // WIDTH)
        value = tl.load(input_ptr + offset, mask, other=0.0)
        diff = value - mean
        variance = tl.where(mask, variance + diff * diff, variance)
    result = tl.sum(variance, 1) / rnumel
    tl.store(output_ptr + tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK), result,
             tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK) < xnumel)


@pytest.mark.parametrize("rnumel", [16, 32, 48, 64])
def test_mixed_axis_reduction_tail(rnumel):
    # XBLOCK=3 crosses WIDTH=8 boundaries and leaves one row in the final tile.
    torch.manual_seed(0)
    source = torch.randn((2, rnumel, 8), dtype=torch.float32)
    expected = source.var(dim=1, correction=0).flatten()
    output = torch.empty(16, dtype=torch.float32, device="npu")
    _mixed_axis_variance[(triton.cdiv(16, 3), )](source.npu(), output, 16, rnumel, 8, 3, 32, multibuffer=False)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-5, atol=1e-5)


@triton.jit
def _two_structured_axes(input_ptr, output_ptr, slo, shi, rlo, rhi, rstride):
    x = tl.arange(0, 3)[:, None, None]
    s = tl.arange(0, 4)[None, :, None]
    r = tl.arange(0, 32)[None, None, :]
    ptr = input_ptr + x % 2 + (s - slo) * rstride * 16 + (r - rlo) * 16
    mask = (s >= slo) & (s < shi) & (r >= rlo) & (r < rhi)
    value = tl.load(ptr, mask, other=-7.0)
    tl.store(output_ptr + x * 128 + s * 32 + r, value)


@pytest.mark.parametrize("slo,shi,rlo,rhi", [(0, 3, 0, 20), (1, 3, 7, 23), (3, 1, 0, 32)])
def test_two_structured_axes(slo, shi, rlo, rhi):
    ns, nr = max(shi - slo, 1), max(rhi - rlo, 1)
    source = torch.arange(ns * nr * 16, dtype=torch.float32).reshape(ns, nr, 16)
    expected = torch.full((3, 4, 32), -7.0)
    if shi > slo and rhi > rlo:
        for row in range(3):
            expected[row, slo:shi, rlo:rhi] = source[:, :, row % 2]
    output = torch.empty_like(expected, device="npu")
    _two_structured_axes[(1, )](source.npu(), output, slo, shi, rlo, rhi, nr, multibuffer=False)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _mixed_axis_column_loop(input_ptr, output_ptr, rows, width, out_stride, TENSOR_OTHER: tl.constexpr):
    x = tl.program_id(0) * 32 + tl.arange(0, 32)[:, None]
    rbase = tl.arange(0, 64)[None, :]
    for start in range(0, width, 64):
        r = start + rbase
        # Irregular rows, contiguous columns and a runtime column tail. The
        # row predicate is loop-covered; the column predicate must survive.
        mask = (x < rows) & (r < width)
        other = (x * 1000 + r).to(tl.float32) if TENSOR_OTHER else -7.0
        value = tl.load(input_ptr + (x % 5) * width + r, mask, other=other)
        tl.store(output_ptr + x * out_stride + r, value)


@pytest.mark.parametrize("width", [63, 64, 65, 130, 512])
@pytest.mark.parametrize("rows", [0, 1, 33])
@pytest.mark.parametrize("tensor_other", [False, True])
@pytest.mark.parametrize("multibuffer", [False, True])
def test_mixed_axis_column_loop(width, rows, tensor_other, multibuffer):
    nr, nc = max(triton.cdiv(rows, 32), 1) * 32, triton.cdiv(width, 64) * 64
    source = torch.arange(5 * width, dtype=torch.float32).reshape(5, width)
    x, r = torch.arange(nr)[:, None], torch.arange(nc)[None, :]
    expected = (x * 1000 + r).float() if tensor_other else torch.full((nr, nc), -7.0)
    expected[:rows, :width] = source[torch.arange(rows) % 5]
    output = torch.empty((nr, nc), dtype=torch.float32, device="npu")
    _mixed_axis_column_loop[(nr // 32, )](source.npu(), output, rows, width, nc, tensor_other, multibuffer=multibuffer)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _mixed_axis_row_gate(input_ptr, output_ptr, lower, upper, width, gate, EQUALITY: tl.constexpr):
    x = tl.arange(0, 8)[:, None]
    r = tl.arange(0, 32)[None, :]
    row_mask = (x == lower) if EQUALITY else ((x >= lower) & (x < upper))
    mask = row_mask & (r < width) & gate
    value = tl.load(input_ptr + (x % 2) * width + r, mask, other=-7.0)
    tl.store(output_ptr + x * 32 + r, value)


@pytest.mark.parametrize("lower,upper,equality", [(-5, 3, False), (2, 7, False), (7, 2, False), (-5, 8, True),
                                                  (3, 8, True)])
@pytest.mark.parametrize("gate", [False, True])
def test_mixed_axis_row_gate(lower, upper, equality, gate):
    width = 20
    source = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    x = torch.arange(8)
    valid = ((x == lower) if equality else ((x >= lower) & (x < upper))) & gate
    expected = torch.full((8, 32), -7.0)
    expected[valid, :width] = source[x[valid] % 2]
    output = torch.empty_like(expected, device="npu")
    _mixed_axis_row_gate[(1, )](source.npu(), output, lower, upper, width, gate, equality, multibuffer=False)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
