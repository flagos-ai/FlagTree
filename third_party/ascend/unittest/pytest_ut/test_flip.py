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
import triton.language.extra.cann.extension as al

import torch
import torch_npu
import pytest
import test_common

# ---------------
# test flip op
# ---------------


@triton.jit
def flip_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = al.flip(x, dim=0)
    tl.store(out_ptr + offs, y)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
def test_flip(dtype):
    size = 64
    x = test_common.generate_tensor((size, ), dtype).npu()
    torch_ref = torch.flip(x, dims=[0])
    triton_res = torch.zeros(size, dtype=eval('torch.' + dtype)).npu()
    flip_kernel[(1, )](x, triton_res, size)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)
