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


@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0, L : tl.constexpr, M : tl.constexpr, N : tl.constexpr):
    lblk_idx = tl.arange(0,L)
    mblk_idx = tl.arange(0,M)
    nblk_idx = tl.arange(0,N)
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0=tl.load(in_ptr0+idx)
    x1=tl.load(in_ptr1+idx)
    ret = x0 + x1
    for i in tl.range(2,5,2):
        ret = ret + x1
    for i in tl.static_range(2,10,3):
        ret = ret + x0
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0+odx, ret)

testlist = [
    (3,5,8),
]

def get_torch_typename(dtype):
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

typelist = ['int8','int16','int32','int64']

@pytest.mark.parametrize('L, M, N',testlist)
@pytest.mark.parametrize('sigtype',typelist)
def test_add(sigtype, L, M, N):
    dtype = get_torch_typename(sigtype)
    shape = (L, M, N)
    x0 = test_common.generate_tensor(shape = (L, M, N),dtype = sigtype).npu()
    x1 = test_common.generate_tensor(shape = (L, M, N),dtype = sigtype).npu()
    y_ref = x0 + x1 + x1 + x1 + x0 + x0 + x0
    output = torch.zeros(shape, dtype=dtype).npu()
    triton_add[1, 1, 1](x0, x1, output, L, M, N)
    test_common.validate_cmp(sigtype, output, y_ref)