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

import time
import torch
import torch_npu
import test_common

NBLOCKS = 1
X_SIZE : tl.constexpr = 4
Y_SIZE : tl.constexpr = 64
Z_SIZE : tl.constexpr = 32
NUMEL = X_SIZE * Y_SIZE * Z_SIZE

def fn(input):
    output = input.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute((1, 0, 2)).reshape((X_SIZE * Y_SIZE * Z_SIZE))
    return output

@triton.jit
def fn_kernel(output_ptr, input_ptr):
    col_offsets = tl.arange(0, X_SIZE * Y_SIZE * Z_SIZE)
    input_local = tl.load(input_ptr + col_offsets)
    input_local = input_local.reshape((X_SIZE, Y_SIZE, Z_SIZE)).permute((1, 0, 2)).reshape((X_SIZE * Y_SIZE * Z_SIZE))  
    tl.store(output_ptr + col_offsets, input_local)


def test_cases():
    input = torch.randn(NUMEL, dtype=torch.float16).npu()
    output = torch.randn(NUMEL, dtype=torch.float16).npu()
    output2 = torch.randn(NUMEL, dtype=torch.float16).npu()
    fn_kernel[1,1,1](output, input)
    output2 = fn(input)
    test_common.validate_cmp('float16', output, output2)
    print("data validation passed")
