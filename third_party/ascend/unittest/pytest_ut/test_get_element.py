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
# test get_element op
# ---------------


@triton.jit
def get_element_kernel(x_ptr, out_ptr, N: tl.constexpr, IDX: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    v = al.get_element(x, [IDX])
    tl.store(out_ptr, v)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
def test_get_element(dtype):
    size, idx = 64, 10
    x = test_common.generate_tensor((size, ), dtype).npu()
    out = torch.zeros(1, dtype=eval('torch.' + dtype)).npu()
    get_element_kernel[(1, )](x, out, size, idx)
    assert out.item() == x[idx].item(), (out.item(), x[idx].item())
