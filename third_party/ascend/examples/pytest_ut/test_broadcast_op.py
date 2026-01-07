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
import torch_npu

import pytest

import test_common

NBLOCKS = 1
XS : tl.constexpr = 2
YS : tl.constexpr = 4
ZS : tl.constexpr = 8
NUMEL : tl.constexpr = XS * ZS

@triton.jit
def fn_broadcast_to(output_ptr, input_ptr, length):
    col_offsets = tl.arange(0, NUMEL)
    input = tl.load(input_ptr + col_offsets)
    result = input.reshape((XS, 1, ZS)).broadcast_to((XS, YS, ZS)).reshape((XS * YS * ZS))
    brc_col_offsets = tl.arange(0, NUMEL * YS)
    tl.store(output_ptr + brc_col_offsets, result)




@pytest.mark.parametrize('dtype', ["float32"])
def test_broadcast_to(dtype):
    length = NUMEL
    dtype = eval(f"torch.{dtype}")
    x = torch.randn((XS, 1, ZS), dtype=dtype).npu()
    output = torch.randn((XS, YS, ZS), dtype=dtype).npu()
    fn_broadcast_to[NBLOCKS,1,1](output, x, length, debug=True)
    assert(torch.equal(output, x.repeat(1, YS, 1)))
