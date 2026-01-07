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


def standard_i2f_float32(x0):
    res = x0.to(torch.float32)
    return res


def standard_i2f_float16(x0):
    res = x0.to(torch.float16)
    return res


def standard_i2f_bfloat16(x0):
    res = x0.to(torch.bfloat16)
    return res


@triton.jit
def triton_i2f_float32(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.float32)
    tl.store(out_ptr0 + idx_block, res, mask=mask)


@triton.jit
def triton_i2f_float16(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.float16)
    tl.store(out_ptr0 + idx_block, res, mask=mask)


@triton.jit
def triton_i2f_bfloat16(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    mask = idx_block < N
    x = tl.load(in_ptr0 + idx_block, mask=mask)
    res = tl.cast(x, tl.bfloat16)
    tl.store(out_ptr0 + idx_block, res, mask=mask)


types = [
    # (torch.int8, 'int8'),   # TO BE FIXED i8 -> f16、bf16
    # (torch.int16, 'int16'), # TO BE FIXED i16 -> f32、bf16
    (torch.int32, 'int32'), # TO BE FIXED i32 -> f16、bf16
    # (torch.int64, 'int64'), # TO BE FIXED i64 -> bf16
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes = [
    (3, 32),
]

map_for_64_t = {37: 31}

ops = [
    # ('i2f16', triton_i2f_float16, standard_i2f_float16, 'float16'),
    ('i2f32', triton_i2f_float32, standard_i2f_float32, 'float32'),
    # ('i2fbf16', triton_i2f_bfloat16, standard_i2f_bfloat16, 'bfloat16'),
]


def continue_func(opName, d_type):
    if 'i2f' in opName and 'float' in d_type:
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
