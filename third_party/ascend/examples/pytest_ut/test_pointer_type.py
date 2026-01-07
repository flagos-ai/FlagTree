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
def kernel(ans_ptr, x_ptr):
    val = tl.load(x_ptr)
    output_ptr = tl.load(ans_ptr)
    output_ptr = output_ptr.to(tl.pointer_type(val.dtype))
    tl.store(output_ptr, val)

@pytest.mark.parametrize("literal, dtype_str",[[0, eval('torch.int8')], [0, eval('torch.int16')],
                                               [0, eval('torch.int32')], [0, eval('torch.int64')],
                                               [0, eval('torch.float16')], [0, eval('torch.float32')]])
def test_pointer_type(literal, dtype_str):
    x = torch.randint(low=0, high=5, size=(1,), dtype=dtype_str).npu()
    output = torch.zeros((1,), dtype=dtype_str).npu()
    ans = []
    ans.append(output.data_ptr())
    ans_tensor = torch.tensor(ans).npu()
    kernel[(1,)](ans_tensor, x)
    assert torch.isclose(x, output)
    print("Pointer type convert successful")