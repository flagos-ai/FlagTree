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

import pytest

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import test_common

import torch
import torch_npu


@triton.jit
def triton_fn_expanddims(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr, YBLOCK_SUB: tl.constexpr):
    base1 = tl.arange(0, XBLOCK)[:, None]
    base2 = tl.arange(0, YBLOCK_SUB)[None, :]
    loops1: tl.constexpr = YBLOCK // YBLOCK_SUB  # assume it's divisible
    for _ in range(loops1):
        x0 = base1 * YBLOCK + base2
        base2 = base2 + YBLOCK_SUB
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tl.store(out_ptr0 + (x0), tmp0, None)


@triton.jit
def triton_fn_broadcast(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr, YBLOCK_SUB: tl.constexpr):
    base1 = tl.arange(0, XBLOCK)[:, None]
    base2 = tl.arange(0, YBLOCK_SUB)[None, :]
    base2 = base2.broadcast_to((XBLOCK, YBLOCK_SUB))
    loops1: tl.constexpr = YBLOCK // YBLOCK_SUB  # assume it's divisible
    for _ in range(loops1):
        x0 = base1 * YBLOCK + base2
        base2 = base2 + YBLOCK_SUB
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tl.store(out_ptr0 + (x0), tmp0, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (128, 128), 128, 128, 32],
])
def test_expanddims(param_list):
    dtype, shape, xblock, yblock, yblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()

    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_fn_expanddims[(1, )](x0, y_cal, xblock, yblock, yblock_sub)
    test_common.validate_cmp(dtype, y_cal, x0)


@pytest.mark.parametrize('param_list', [
    ['float32', (128, 128), 128, 128, 32],
])
def test_broadcast(param_list):
    dtype, shape, xblock, yblock, yblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()

    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_fn_broadcast[(1, )](x0, y_cal, xblock, yblock, yblock_sub)
    test_common.validate_cmp(dtype, y_cal, x0)
