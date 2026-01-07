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


@triton.jit
def fn_npu_(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    i = tl.arange(0, 4)
    j = tl.arange(0, 4)
    xx, yy = tl.swizzle2d(i, j, size_i=4, size_j=4, size_g=2)

    tl.store(output_ptr + tl.arange(0, 4), xx)
    tl.store(x_ptr + tl.arange(0, 4), yy)


@pytest.mark.parametrize('param_list',
                         [
                             ['int32', (2, 256, 16), 1, 2, 256, 16],
                         ]
                         )
def test_case(param_list):
    dtype, shape, ncore, XB, YB, ZB = param_list
    x = test_common.generate_tensor((4,), dtype).npu()
    a = torch.tensor([[0, 2, 1, 3],
                      [4, 6, 5, 7],
                      [8, 10, 9, 11],
                      [12, 14, 13, 15]], dtype=eval('torch.' + dtype)).npu()
    output = torch.randint(1, (4,), dtype=eval('torch.' + dtype)).npu()
    fn_npu_[ncore, 1, 1](output, x, XB, YB, ZB)
    print(f"output={output}")
    triton_ret = output[:, None] * 4 + x[None, :]
    print(f"triton_ret={triton_ret}")
    torch.testing.assert_close(triton_ret, a)
