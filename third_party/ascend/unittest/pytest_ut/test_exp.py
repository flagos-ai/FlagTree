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
import numpy as np
import torch
import pytest
import test_common


def torch_pointwise(x0):
    res = torch.exp(x0)
    return res


@triton.jit
def triton_exp(in_ptr0, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp2 = tl.exp(tmp0)
        tl.store(out_ptr0 + (x0), tmp2, None)


@pytest.mark.parametrize('param_list', [
    ['float32', (2, 4096, 8), 2, 32768, 1024],
])
def test_case(param_list):
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = torch_pointwise(x0)
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_exp[ncore, 1, 1](x0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@triton.jit
def triton_elementwise_unary(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N)
    ret = tl.exp(x)
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < N)


shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]


@pytest.mark.parametrize('dtype, sigtype', [(torch.float32, 'float32'), (torch.float16, 'float16')])
@pytest.mark.parametrize('N, NUMEL', shapes)
def test_elementwsie_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    x0 = test_common.generate_tensor(shape=(N, ), dtype=sigtype)

    ans = torch_pointwise(x0)
    x0 = x0.npu()

    out = torch.zeros((N, ), dtype=dtype).npu()
    triton_elementwise_unary[1, 1, 1](x0, out, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(sigtype, out, ans)
