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


@triton.jit
def fn_npu_3d(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    block_ptr_in = tl.make_block_ptr(base=x_ptr, shape=(XB, YB, ZB), strides=(YB * ZB, ZB, 1), offsets=(0, 0, 0),
                                     block_shape=(XB, YB, 2), order=(2, 1, 0))
    block_ptr_out = tl.make_block_ptr(base=output_ptr, shape=(XB, YB, ZB), strides=(YB * ZB, ZB, 1), offsets=(0, 0, 0),
                                      block_shape=(XB, YB, 2), order=(2, 1, 0))
    pid = tl.program_id(axis=0)  # pid=0,1  BLOCK_SIZE_N=8
    for _ in range(ZB // 2):
        X = tl.load(block_ptr_in, boundary_check=(0, 1, 2))
        tl.store(block_ptr_out, X, boundary_check=(0, 1, 2))
        block_ptr_in = tl.advance(block_ptr_in, (0, 0, 2))
        block_ptr_out = tl.advance(block_ptr_out, (0, 0, 2))


@pytest.mark.parametrize('dtype', ["int32", "float32", "int16"])
@pytest.mark.parametrize('shape', [(33, 9, 6), (8, 8, 4)])
def test_advance_with_boundary_check(dtype, shape):
    x = torch.randint(low=-128, high=128, size=shape, dtype=eval('torch.' + dtype)).npu()
    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    expected = x
    output = torch.randint(1, shape, dtype=eval('torch.' + dtype)).npu()
    fn_npu_3d[1, 1, 1](output, x, XB=shape[0], YB=shape[1], ZB=shape[2])
    torch.testing.assert_close(output, expected)
