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
def triton_sink_broadcast1(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    base1 = tl.arange(0, XBLOCK)[:, None] * YBLOCK
    base2 = tl.arange(0, YBLOCK)[None, :]
    base1 = base1.broadcast_to((XBLOCK, YBLOCK))
    tmp0 = tl.load(in_ptr0 + base1, None)
    index = base1 + base2
    tl.store(out_ptr0 + index, tmp0, None)


@triton.jit
def triton_sink_broadcast2(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    base1 = tl.arange(0, XBLOCK)[:, None] * YBLOCK
    base2 = tl.arange(0, YBLOCK)[None, :]
    base1 = base1.broadcast_to((XBLOCK, YBLOCK))
    tmp0 = tl.load(in_ptr0 + base1, base1 < XBLOCK * YBLOCK, other=0.0)
    index = base1 + base2
    tl.store(out_ptr0 + index, tmp0, index < XBLOCK * YBLOCK)


@triton.jit
def triton_sink_broadcast3(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    base1 = (tl.arange(0, XBLOCK) * YBLOCK)[:, None]
    base2 = tl.arange(0, YBLOCK)[None, :]
    base1 = base1.broadcast_to((XBLOCK, YBLOCK))
    tmp0 = tl.load(in_ptr0 + base1 + base2, (base1 + base2) < XBLOCK * YBLOCK, other=0.0)
    index = base1 + base2
    tl.store(out_ptr0 + index, tmp0, index < XBLOCK * YBLOCK)


@pytest.mark.parametrize('param_list', [
    ['float32', (32, 32), 32, 32],
])
def test_sink_broadcast(param_list):
    dtype, shape, xblock, yblock = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0.clone()
    y_ref = y_ref[:, 0].unsqueeze(1).expand(-1, x0.size(1))

    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    y_cal2 = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_sink_broadcast1[(1, )](x0, y_cal, xblock, yblock)
    triton_sink_broadcast2[(1, )](x0, y_cal2, xblock, yblock)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    test_common.validate_cmp(dtype, y_cal2, y_ref)


@pytest.mark.parametrize('param_list', [
    ['float32', (32, 32), 32, 32],
])
def test_sink_broadcast3(param_list):
    dtype, shape, xblock, yblock = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0.clone()

    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_sink_broadcast3[(1, )](x0, y_cal, xblock, yblock)
    test_common.validate_cmp(dtype, y_cal, y_ref)
