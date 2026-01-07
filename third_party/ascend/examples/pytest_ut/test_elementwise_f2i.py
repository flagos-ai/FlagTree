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
import test_common
import os
import shutil

import torch
import torch_npu

def standard_f2i32(x0):
    res = x0.to(torch.int32)
    return res

def standard_f2i8(x0):
    res = x0.to(torch.int8)
    return res

def standard_f2i16(x0):
    res = x0.to(torch.int16)
    return res

def standard_f2i64(x0):
    res = x0.to(torch.int64)
    return res

@triton.jit
def triton_f2i8(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.int8)
    tl.store(out_ptr0 + idx_block, res, mask=mask)

@triton.jit
def triton_f2i16(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.int16)
    tl.store(out_ptr0 + idx_block, res, mask=mask)

@triton.jit
def triton_f2i32(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.int32)
    tl.store(out_ptr0 + idx_block, res, mask=mask)

@triton.jit
def triton_f2i64(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.int64)
    tl.store(out_ptr0 + idx_block, res, mask=mask)

types = [
    (torch.float32, 'float32'),
    (torch.float16, 'float16'),
    (torch.bfloat16, 'bfloat16'),
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes = [
    (3, 32),
]

map_for_64_t = {37: 31}

ops = [
    ('f2i8', triton_f2i8, standard_f2i8, 'int8'),
    ('f2i16', triton_f2i16, standard_f2i16, 'int16'),
    ('f2i32', triton_f2i32, standard_f2i32, 'int32'),
    ('f2i64', triton_f2i64, standard_f2i64, 'int64'),
]


def continue_func(opName, d_type):
    if 'f2i' in opName and 'int' in d_type:
        return True


@pytest.mark.parametrize('opName, tritonOp, standOp, dst_sigtype', ops)
@pytest.mark.parametrize('dtype, sigtype', types)
@pytest.mark.parametrize('N, NUMEL', shapes)
def test_elementwise_common(opName, tritonOp, standOp, dst_sigtype, dtype, sigtype, N, NUMEL):
    if continue_func(opName, sigtype):
        return

    torch_npu.npu.utils.set_device(0)
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N] if N in map_for_64_t else N

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)

    ans = standOp(x0)
    x0 = x0.npu()

    output = test_common.generate_tensor(shape=(N,), dtype=dst_sigtype).npu()
    tritonOp[1, 1, 1](x0, output, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(dst_sigtype, output, ans)
