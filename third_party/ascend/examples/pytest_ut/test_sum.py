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
import time


@triton.jit
def sum_loop_high(in_ptr0, in_ptr1, in_ptr2, out_ptr0, rnumel, xnumel,
                  XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    R = rnumel
    X = xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xbase = tl.arange(0, XBLOCK_SUB)
    rbase = tl.arange(0, RBLOCK)
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + xbase
        x0 = xindex[None, :]
        _tmp6 = tl.full([RBLOCK, XBLOCK_SUB], 0, tl.float32)
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = None
            r1 = rindex[:, None]
            tmp0 = tl.load(in_ptr0 + (X * r1 + (x0)), rmask)
            tmp1 = tl.load(in_ptr1 + (X * r1 + (x0)), rmask)
            tmp3 = tl.load(in_ptr2 + (X * r1 + (x0)), rmask)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            _tmp6 = _tmp6 + tmp4
        tmp6 = tl.sum(_tmp6, 0)
        tl.store(out_ptr0 + (xindex), tmp6, None)


@triton.jit
def sum_loop_low(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ynumel,
                 XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    X = xnumel
    Y = ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)

    x0 = xindex[:, None]
    rbase = tl.arange(0, RBLOCK)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, ynumel, RBLOCK):
        rindex = roffset + rbase
        rmask = None
        r1 = rindex[None, :]
        tmp0 = tl.load(in_ptr0 + (r1 + (Y * x0)), rmask)
        tmp1 = tl.load(in_ptr1 + (r1 + (Y * x0)), rmask)
        tmp3 = tl.load(in_ptr2 + (r1 + (Y * x0)), rmask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        _tmp6 = _tmp6 + tmp4
    tmp6 = tl.sum(_tmp6, 1)

    tl.store(out_ptr0 + (xindex), tmp6, None)


def foo(a, b, c):
    y = a + b + c
    y = y.sum(0)
    return y


def bar(a, b, c):
    y = a + b + c
    y = y.sum(1)
    return y


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (64, 8192), 1, 2, 256, 16],
                         ]
                         )
def test_case_1(param_list):
    dtype, shape, ncore, XB, YB, ZB = param_list
    a = test_common.generate_tensor(shape, dtype).npu()
    b = test_common.generate_tensor(shape, dtype).npu()
    c = test_common.generate_tensor(shape, dtype).npu()
    value = torch.empty_strided((a.shape[0],), (1,)).npu()

    std_low_ret = bar(a, b, c)
    print(f"std_low_ret = {std_low_ret[0:8]}")
    XBLOCK = 64
    RBLOCK = 32
    NBLOCKS = a.shape[0] // XBLOCK
    sum_loop_low[NBLOCKS, 1, 1](a, b, c, value, a.shape[0], a.shape[1], XBLOCK, RBLOCK)
    triton_low_ret = value
    print(f"triton_low_ret = {triton_low_ret[0:8]}")
    torch.testing.assert_close(std_low_ret, triton_low_ret, rtol=1e-3, atol=1e-3)

    std_ret2 = foo(a, b, c)
    print(f"std_ret2 = {std_ret2[0:8]}")
    NBLOCKS = 32
    XBLOCK = a.shape[1] // NBLOCKS
    XBLOCK_SUB = min(64, max(XBLOCK // 2, 32))
    RBLOCK = 64

    value2 = torch.empty_strided((a.shape[1],), (1,)).npu()
    sum_loop_high[NBLOCKS, 1, 1](a, b, c, value2, a.shape[0], a.shape[1], XBLOCK, XBLOCK_SUB, RBLOCK)
    triton_ret2 = value2
    print(f"triton_ret2 = {triton_ret2[0:8]}")
    torch.testing.assert_close(std_ret2, triton_ret2, rtol=1e-3, atol=1e-3)
