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
"""Regression for common-base pointer reshape across masked loads and stores."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

# Remove once NPUIR fixes arange_2d S/V synchronization and packed i1 scalar
# extraction on A2/A3. The offset error can also leave later tests failing with
# device exceptions, so skip the entire family until both backend fixes land.
pytestmark = pytest.mark.skipif(
    triton.runtime.driver.active.get_current_target().arch.startswith(("Ascend910B", "Ascend910_93")),
    reason="NPUIR A2/A3: arange_2d synchronization and packed-mask scalar extraction are incorrect",
)


@triton.jit
def _masked_pointer_reshape(src, gathered, masked_out, n, WIDE: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    logical = (lanes * 7) % BLOCK
    offsets = logical - BLOCK // 2
    if WIDE:
        offsets = offsets.to(tl.int64)
    offsets_2d = tl.reshape(offsets, (4, BLOCK // 4))
    # The scalar base displacement and negative per-lane offsets must cancel.
    src_ptrs = (src + BLOCK // 2) + offsets_2d
    src_flat = tl.reshape(src_ptrs, (BLOCK, ))
    valid = logical < n
    values = tl.load(src_flat, mask=valid, other=-7)
    tl.store(gathered + lanes, values)

    dst_ptrs = masked_out + lanes
    dst_2d = tl.reshape(dst_ptrs, (4, BLOCK // 4))
    tl.store(dst_2d, tl.reshape(values, (4, BLOCK // 4)), mask=tl.reshape(valid, (4, BLOCK // 4)))


@pytest.mark.parametrize("n", [0, 17, 32])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_masked_pointer_reshape(n, wide, dtype):
    block = 32
    # Masked-off source lanes contain distinct poison values, and masked-off
    # output lanes must keep the sentinel rather than storing the load's other.
    src_cpu = torch.arange(block, dtype=dtype) * 3 + 5
    src_cpu[n:] = -1000
    src = src_cpu.npu()
    gathered = torch.empty(block, dtype=dtype, device="npu")
    masked_out = torch.full((block, ), 123, dtype=dtype, device="npu")
    _masked_pointer_reshape[(1, )](src, gathered, masked_out, n, WIDE=wide, BLOCK=block)

    indices = (torch.arange(block) * 7) % block
    valid = indices < n
    expected = torch.where(valid, src_cpu[indices], -7)
    expected_out = torch.where(valid, expected, 123)
    torch.testing.assert_close(gathered.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(masked_out.cpu(), expected_out, rtol=0, atol=0)


@triton.jit
def _common_base_reshape_variants(src, out, shift, steps, ROW_STRIDE: tl.constexpr, BROADCAST: tl.constexpr):
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 8)
    if BROADCAST:
        ptrs = (src + shift) + rows[:, None].to(tl.int64)
        ptrs = tl.broadcast_to(ptrs, (4, 8))
    else:
        offsets = rows[:, None] * ROW_STRIDE + cols[None, :]
        ptrs = (src + shift) + offsets.to(tl.int64)
    ptrs = ptrs + 2
    for _ in range(steps):
        ptrs = ptrs + 1
    flat = tl.reshape(ptrs, (32, ))
    values = tl.load(flat)
    dst = tl.reshape(out + tl.arange(0, 32), (2, 16))
    tl.store(dst, tl.reshape(values, (2, 16)))


@pytest.mark.parametrize("row_stride,broadcast", [(8, False), (16, False), (0, True)])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_common_base_reshape_variants(row_stride, broadcast, steps, dtype):
    shift = 7
    src_cpu = torch.arange(80, dtype=dtype) * 3 + 5
    src = src_cpu.npu()
    out = torch.empty(32, dtype=dtype, device="npu")
    _common_base_reshape_variants[(1, )](src, out, shift, steps, ROW_STRIDE=row_stride, BROADCAST=broadcast)
    rows = torch.arange(4)[:, None]
    if broadcast:
        offsets = rows.expand(4, 8)
    else:
        offsets = rows * row_stride + torch.arange(8)[None, :]
    indices = (offsets + shift + 2 + steps).reshape(-1)
    torch.testing.assert_close(out.cpu(), src_cpu[indices], rtol=0, atol=0)
