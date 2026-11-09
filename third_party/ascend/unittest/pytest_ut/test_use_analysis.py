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
def triton_reduce_deadcode(v_ptr, in_ptr0, in_ptr1, out_ptr0, VBLOCK: tl.constexpr, XBLOCK: tl.constexpr,
                           YBLOCK: tl.constexpr):
    v_idx = tl.arange(0, VBLOCK)
    v = tl.load(v_ptr + v_idx)
    v_ret = tl.argmax(v, 0)
    if v_ret < v_ret + 1:
        for _ in range(v_ret, v_ret + 1):
            cube_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]
            c0 = tl.load(in_ptr0 + cube_idx)
            c1 = tl.load(in_ptr1 + cube_idx)
            ret = tl.dot(c0, c1) + 1
            tl.store(out_ptr0 + cube_idx, ret)
    else:
        for _ in range(v_ret - 1, v_ret):
            cube_idx = tl.arange(0, XBLOCK)[:, None] * YBLOCK + tl.arange(0, YBLOCK)[None, :]
            c0 = tl.load(in_ptr0 + cube_idx)
            c1 = tl.load(in_ptr1 + cube_idx)
            ret = tl.dot(c0, c1) + 1
            tl.store(out_ptr0 + cube_idx, ret)


def torch_reduce_deadcode(in0, in1, v):
    v_ret = torch.argmax(v)
    if v_ret < v_ret + 1:
        ret = torch.matmul(in0, in1) + 1
    else:
        ret = torch.matmul(in0, in1) + 1
    return ret


def test_reduce_deadcode():
    VBLOCK, XBLOCK, YBLOCK = 16, 16, 16
    sigtype = 'float32'
    dtype = torch.float32
    in0 = torch.randn((XBLOCK, YBLOCK), dtype=dtype, device='npu')
    in1 = torch.randn((XBLOCK, YBLOCK), dtype=dtype, device='npu')
    v = torch.randn((VBLOCK, ), dtype=dtype, device='npu')
    out = torch.zeros((XBLOCK, YBLOCK), dtype=dtype, device='npu')

    triton_reduce_deadcode[(1, )](v, in0, in1, out, VBLOCK=VBLOCK, XBLOCK=XBLOCK, YBLOCK=YBLOCK)
    expected = torch_reduce_deadcode(in0, in1, v)
    test_common.validate_cmp(sigtype, out, expected)
