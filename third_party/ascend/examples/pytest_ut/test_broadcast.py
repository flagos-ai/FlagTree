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
def fn_broadcast(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = tl.arange(0, 1)[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    x1 = tl.load(out_ptr0 + odx)
    ret = tl.broadcast(x, x1)
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', ["bfloat16"])
def test_broadcast_alltype(dtype):
    input = test_common.generate_tensor((1, YS, ZS), dtype).npu()
    ans = input.repeat(XS, 1, 1)
    output = torch.zeros((XS, YS, ZS), dtype=eval('torch.' + dtype)).npu()
    fn_broadcast[1, 1, 1](input, output, XS, YS, ZS)
    test_common.validate_cmp(dtype, ans, output)
    