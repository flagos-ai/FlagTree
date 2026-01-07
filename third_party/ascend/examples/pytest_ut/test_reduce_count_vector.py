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

import torch
import torch_npu


def standard_count(x0, cmp_val, dim):
    res = (x0 == cmp_val).sum(dim=dim)
    return res


def standard_gt(x0, cmp_val, dim):
    res = (x0 > cmp_val).sum(dim=dim)
    return res


def standard_lt(x0, cmp_val, dim):
    res = (x0 < cmp_val).sum(dim=dim)
    return res


@triton.jit
def triton_count(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx_block)

    tmp3 = (x == cmp_val)
    # tmp3 bool -> tl.float32
    tmp4 = tmp3.to(tl.float32)
    res = tl.sum(tmp4, dim)

    tl.store(out_ptr0 + idx_block, res)


@triton.jit
def triton_gt(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx_block)

    tmp3 = (x > cmp_val)
    # tmp3 bool -> tl.float32
    tmp4 = tmp3.to(tl.float32)
    res = tl.sum(tmp4, dim)

    tl.store(out_ptr0 + idx_block, res)


@triton.jit
def triton_lt(in_ptr0, out_ptr0, cmp_val, dim: tl.constexpr, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx_block)

    tmp3 = (x < cmp_val)
    # tmp3 bool -> tl.float32
    tmp4 = tmp3.to(tl.float32)
    res = tl.sum(tmp4, dim)

    tl.store(out_ptr0 + idx_block, res)


types = [
    (torch.float32, 'float32'),
    (torch.float16, 'float16'),
    (torch.bfloat16, 'bfloat16'),
    (torch.int8, 'int8'),
    (torch.int16, 'int16'),
    (torch.int32, 'int32'),
    (torch.int64, 'int64'),
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes = [
    (32, 32),
]

map_for_64_t = {37: 31}

CPM_VAL_INT = 8
CPM_VAL_FLOAT = 0.5

# TO BE FIXED with mask
ops = [
    ('counti', triton_count, standard_count, CPM_VAL_INT),
    ('countf', triton_gt, standard_gt, CPM_VAL_FLOAT),
    ('countf', triton_lt, standard_lt, CPM_VAL_FLOAT),
]


def judge_continue(opName, sigtype):
    if opName == 'counti' and 'int' in sigtype:
        return False
    if opName == 'countf' and 'float' in sigtype:
        return False
    return True


@pytest.mark.parametrize('opName, tritonOp, standOp, cmp_val', ops)
@pytest.mark.parametrize('dtype, sigtype', types)
@pytest.mark.parametrize('N, NUMEL', shapes)
def test_reduce_count_vector(opName, tritonOp, standOp, cmp_val, dtype, sigtype, N, NUMEL):
    if judge_continue(opName, sigtype):
        return
    torch.manual_seed(0)
    torch_npu.npu.utils.set_device(0)
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N] if N in map_for_64_t else N

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype)
    ans = standOp(x0, cmp_val, 0)
    x0 = x0.npu()

    output = torch.tensor(0, dtype=torch.float32).npu()
    tritonOp[1, 1, 1](x0, output, cmp_val, dim=0, N=N, NUMEL=NUMEL, debug=True)
    output = output.cpu().to(torch.int32)
    # print(f'x0:{x0}\ntriton:{output}\ntorch:{ans}')
    assert torch.equal(output, ans)


if __name__ == "__main__":
    dtype = torch.float32
    sigtype = 'float32'
    allshape = [(3, 32)]
    for shape in allshape:
        test_reduce_count_vector('countf', triton_lt, standard_lt, CPM_VAL_FLOAT, dtype, sigtype, shape[0], shape[1])
