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

import torch
import torch_npu

def fn(x):
    return x.t()

@triton.jit
def triton_2d_permute(output_ptr, input_ptr, X : tl.constexpr, Y : tl.constexpr):
    xindex = tl.arange(0, X * Y)
    input_local = tl.load(input_ptr + xindex)
    output_local = input_local.reshape(X, Y).trans().reshape(X*Y)
    tl.store(output_ptr + xindex, output_local)


@pytest.mark.parametrize('X', [32, 64, 256])
@pytest.mark.parametrize('Y', [16, 32])
def test_cases(X, Y):

    x = torch.randn((X, Y)).npu()
    output1 = fn(x)
    output2 = torch.randn(output1.shape, dtype=output1.dtype).npu()

    triton_2d_permute[1, 1, 1](output2, x, X, Y, debug=True)
    print(output1)
    print(output2)

    torch.testing.assert_close(output1, output2, rtol=1e-3, atol=1e-3)






