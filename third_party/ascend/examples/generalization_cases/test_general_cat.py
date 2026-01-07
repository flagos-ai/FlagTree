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
from test_common import TestUtils
import math


@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, XB: tl.constexpr):

    idx = tl.arange(0, XB)

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = tl.cat(X, Y, can_reorder=True)

    oidx = tl.arange(0, XB * 2)

    tl.store(output_ptr + oidx, ret)

# The CAT operator in the Triton community also does not support boolean types.
@pytest.mark.parametrize('shape', TestUtils.test_shape1d) #triton only support 1D cat
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8', 'int64'])
def test_cat(shape, dtype):
    m = shape[0]
    x = torch.full((m, ), 100, dtype=eval("torch." + dtype)).npu()
    y = torch.full((m, ), 30, dtype=eval("torch." + dtype)).npu()

    output = torch.randint(1, (m * 2, ), dtype=eval("torch." + dtype)).npu()

    ans = torch.cat((x, y), dim=0)

    fn_npu_[1, 1, 1](output, x, y, m)

    test_common.validate_cmp(dtype, ans, output)

