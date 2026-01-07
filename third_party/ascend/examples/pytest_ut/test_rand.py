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
import pytest
import test_common
import math


@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals) # 存储随机数

@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size, step=4):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint4x(5, 10 + global_offset, n_rounds) # 对每个索引生成一个随机数
        mask = (global_offset + indices) < N
        tl.store(x_ptr + global_offset + indices, rand_vals, mask) # 存储随机数

shapes = [(1,3)]

@pytest.mark.parametrize('shape', shapes)
def test_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)
    kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)

    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)
    kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)
