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


import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math


@triton.jit
def fn_npu_(output_ptr, x_ptr, output_ptr1,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL:tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx=tl.arange(0,XB) + xoffs
    yidx=tl.arange(0,YB) + yoffs
    zidx=tl.arange(0,ZB) + zoffs

    idx=xidx[:,None,None,None]*YNUMEL*ZNUMEL*2+yidx[None,:,None,None]*ZNUMEL*2+ \
         zidx[None,None,:,None]*2 + tl.arange(0,2)[None,None,None,:]

    X = tl.load(x_ptr+idx)

    xx, yy = tl.split(X)

    oidx=xidx[:,None,None]*YNUMEL*ZNUMEL+yidx[None,:,None]*ZNUMEL+zidx[None,None,:]

    tl.store(output_ptr + oidx, xx)
    tl.store(output_ptr1 + oidx, yy)

import logging

@pytest.mark.parametrize('shape', TestUtils.full_shape)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_split(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.'+dtype)).npu()
    y = torch.full(shape, 30, dtype=eval('torch.'+dtype)).npu()
    xx = torch.stack((x, y), dim=-1)

    a, b = torch.split(xx, 1, dim=-1)

    if len(shape) == 1:
        XB = 1;xnumel = 1
        YB = 1;ynumel = 1
        ZB = shape[0];znumel = shape[0]
    elif len(shape) == 2:
        XB = 1;xnumel = 1
        YB = shape[0]; ynumel = shape[0]
        ZB = shape[1];znumel = shape[1]
    else:
        XB = shape[0];xnumel = shape[0]
        YB = shape[1];ynumel = shape[1]
        ZB = shape[2];znumel = shape[2]

    a = a.reshape(XB, YB, ZB)
    b = b.reshape(XB, YB, ZB)
    output = torch.randint(1, (XB,YB,ZB), dtype=eval('torch.'+dtype)).npu()
    output1 = torch.randint(1, (XB,YB,ZB), dtype=eval('torch.'+dtype)).npu()

    grid = (1,1,1)
    if x.numel()*x.element_size() >= 8192:
        if xnumel > 1:
            grid = (XB,1,1)
            XB = 1
        elif ynumel > 1:
            grid = (1,YB,1)
            YB = 1
        else:
            grid = (1,1,ZB)
            ZB = 1

    fn_npu_[grid](output, xx, output1, XB, YB, ZB, xnumel, ynumel, znumel)

    test_common.validate_cmp(dtype, a, output)
    test_common.validate_cmp(dtype, b, output1)


@triton.jit
def fn_npu_4_8d(
    output_ptr, x_ptr, output_ptr1,
    XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
    WB: tl.constexpr, VB: tl.constexpr, UB: tl.constexpr,
    TB: tl.constexpr, SB: tl.constexpr
):

    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)
    widx = tl.arange(0, WB)
    vidx = tl.arange(0, VB)
    uidx = tl.arange(0, UB)
    tidx = tl.arange(0, TB)
    sidx = tl.arange(0, SB)

    idx = (
        xidx[:, None, None, None, None, None, None, None, None] *
        YB * ZB * WB * VB * UB * TB * SB * 2 +
        yidx[None, :, None, None, None, None, None, None, None] *
        ZB * WB * VB * UB * TB * SB * 2 +
        zidx[None, None, :, None, None, None, None, None, None] *
        WB * VB * UB * TB * SB * 2 +
        widx[None, None, None, :, None, None, None, None, None] *
        VB * UB * TB * SB * 2 +
        vidx[None, None, None, None, :, None, None, None, None] *
        UB * TB * SB * 2 +
        uidx[None, None, None, None, None, :, None, None, None] *
        TB * SB * 2 +
        tidx[None, None, None, None, None, None, :, None, None] *
        SB * 2 +
        sidx[None, None, None, None, None, None, None, :, None] * 2 +
        tl.arange(0, 2)[None, None, None, None, None, None, None, None, :]
    )

    X = tl.load(x_ptr + idx)
    xx, yy = tl.split(X)

    oidx = (
        xidx[:, None, None, None, None, None, None, None] *
        YB * ZB * WB * VB * UB * TB * SB +
        yidx[None, :, None, None, None, None, None, None] *
        ZB * WB * VB * UB * TB * SB +
        zidx[None, None, :, None, None, None, None, None] *
        WB * VB * UB * TB * SB +
        widx[None, None, None, :, None, None, None, None] *
        VB * UB * TB * SB +
        vidx[None, None, None, None, :, None, None, None] *
        UB * TB * SB +
        uidx[None, None, None, None, None, :, None, None] *
        TB * SB +
        tidx[None, None, None, None, None, None, :, None] *
        SB +
        sidx[None, None, None, None, None, None, None, :]
    )

    tl.store(output_ptr + oidx, xx)
    tl.store(output_ptr1 + oidx, yy)


@pytest.mark.parametrize('shape', TestUtils.full_shape_4_8d)
@pytest.mark.parametrize('dtype', TestUtils.full_dtype)
def test_split_4_8d(shape, dtype):
    logging.log(logging.DEBUG, f"shape = {shape}")
    x = torch.full(shape, 100, dtype=eval('torch.' + dtype)).npu()
    y = torch.full(shape, 30, dtype=eval('torch.' + dtype)).npu()
    xx = torch.stack((x, y), dim=-1)

    a, b = torch.split(xx, 1, dim=-1)

    if len(shape) == 1:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, 1, 1, 1, 1, 1, shape[0]
    elif len(shape) == 2:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, 1, 1, 1, 1, shape[0], shape[1]
    elif len(shape) == 3:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, 1, 1, 1, shape[0], shape[1], shape[2]
    elif len(shape) == 4:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, 1, 1, shape[0], shape[1], shape[2], shape[3]
    elif len(shape) == 5:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, 1, shape[0], shape[1], shape[2], shape[3], shape[4]
    elif len(shape) == 6:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, 1, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]
    elif len(shape) == 7:
        XB, YB, ZB, WB, VB, UB, TB, SB = 1, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]
    else:
        XB, YB, ZB, WB, VB, UB, TB, SB = shape

    a = a.reshape(XB, YB, ZB, WB, VB, UB, TB, SB)
    b = b.reshape(XB, YB, ZB, WB, VB, UB, TB, SB)

    output = torch.randint(1, (XB, YB, ZB, WB, VB, UB, TB, SB), dtype=eval('torch.' + dtype)).npu()
    output1 = torch.randint(1, (XB, YB, ZB, WB, VB, UB, TB, SB), dtype=eval('torch.' + dtype)).npu()

    grid = (1, 1, 1)
    fn_npu_4_8d[grid](output, xx, output1, XB, YB, ZB, WB, VB, UB, TB, SB)

    test_common.validate_cmp(dtype, a, output)
    test_common.validate_cmp(dtype, b, output1)


@triton.jit
def fn_npu_(output_ptr, x_ptr, output_ptr1, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    xx, yy = tl.split(X)

    oidx = xidx[:, None] * YB + yidx[None, :]

    tl.store(output_ptr + oidx, xx)
    tl.store(output_ptr1 + oidx, yy)

@pytest.mark.parametrize('para_type, data_type, XB, YB, ZB',
                         [
                             ('bfloat16', torch.bfloat16, 2, 8, 2),
                             ('uint8', torch.uint8, 1, 256, 2),
                             ('bool', torch.bool, 1, 1, 2),
                         ])
def test_split_u(para_type, data_type, XB, YB, ZB):
    x = test_common.generate_tensor((XB, YB, ZB), para_type).npu()
    a, b = torch.split(x, 1, dim=-1)
    a = a.reshape(XB, YB)
    b = b.reshape(XB, YB)

    output = test_common.generate_tensor((XB, YB), para_type).npu()
    output1 = test_common.generate_tensor((XB, YB), para_type).npu()
    fn_npu_[1, 1, 1](output, x, output1, XB, YB, ZB, debug=True)

    test_common.validate_cmp(para_type, a, output)
    test_common.validate_cmp(para_type, b, output1)


