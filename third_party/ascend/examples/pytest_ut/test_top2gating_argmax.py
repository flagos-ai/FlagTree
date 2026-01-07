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

import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common


@triton.jit
def promote_to_tensor(x):
    # Addition promotes to tensor for us
    return x + tl.zeros((1,), tl.int1)


@triton.jit
def is_floating(x):
    return promote_to_tensor(x).dtype.is_floating()


@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index):
    mask = a_value > b_value
    equal = a_value == b_value
    if is_floating(a_value):
        a_isnan = a_value != a_value
        b_isnan = b_value != b_value
        mask |= a_isnan and not b_isnan
        # Consider NaNs as equal
        equal |= a_isnan and b_isnan

    # Prefer lowest index if values are equal
    mask |= equal & (a_index < b_index)
    return tl.where(mask, a_value, b_value), tl.where(mask, a_index, b_index)


@triton.jit
def max_with_index(value, index, dim):
    return tl.reduce((value, index), dim, maximum_with_index)


@triton.jit
def triton_4(in_ptr2, in_ptr4, out_ptr10, x0_numel, r1_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    RBLOCK: tl.constexpr = 4
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = r1_numel // RBLOCK
    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        r1 = base2[None, :]
        tmp15 = tl.load(in_ptr2 + (r1 + (8 * x0)), None)
        tmp25 = tl.load(in_ptr4 + (r1 + (4 * x0)), None)
        tmp21 = tmp15.to(tl.float32)
        tmp26 = tmp25 * tmp21
        tmp27 = tl.reshape(tmp26, [XBLOCK_SUB, RBLOCK])
        tmp31 = tl.broadcast_to(r1.reshape(1, RBLOCK), tmp27.shape)
        _, tmp30_tmp = max_with_index(tmp27, tmp31, 1)
        tmp30 = tmp30_tmp.reshape(XBLOCK_SUB, 1)
        tmp43 = tmp30.to(tl.int32)
        tmp46 = tmp43
        tl.store(out_ptr10 + (x0), tmp46, None)


def test_max_with_index_dim0():
    mask = torch.randint(low=0, high=2, size=(512, 2, 4), dtype=torch.int32).npu()
    weights = torch.randn((512, 4), device='npu', dtype=torch.float32)
    buf32 = torch.randint(low=0, high=8, size=(512,), dtype=torch.int32).npu()
    XBLOCK = 32
    XBLOCK_SUB = 32
    triton_4[16, 1, 1](mask, weights, buf32, 512, 4, XBLOCK, XBLOCK_SUB)

    _, first_idx = torch.max(weights * mask[:, 0, :], dim=1)
    print(f"first_idx: {first_idx[0:8]}")
    print(f"triton_idx: {buf32[0:8]}")

    assert torch.all(torch.eq(first_idx, buf32.to(first_idx.dtype)))
