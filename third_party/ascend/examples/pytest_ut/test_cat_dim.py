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
def fn3_dim0(output_ptr, x1_ptr, x2_ptr, x3_ptr, x1_shape: tl.constexpr, x2_shape: tl.constexpr, x3_shape: tl.constexpr):
    idx_start = 0
    x1_idx = tl.arange(0,x1_shape)
    X1 = tl.load(x1_ptr + x1_idx)
    tl.store(output_ptr + x1_idx, X1)

    idx_start += x1_shape
    x2_idx = tl.arange(0,x2_shape)
    X2 = tl.load(x2_ptr + x2_idx)
    tl.store(output_ptr + idx_start + x2_idx, X2)

    idx_start += x2_shape
    x3_idx = tl.arange(0,x3_shape)
    X3 = tl.load(x3_ptr + x3_idx)
    tl.store(output_ptr + idx_start + x3_idx, X3)


@triton.jit
def fn4_dim1(output_ptr, x0_ptr, x1_ptr, x2_ptr, x3_ptr, dim0_len: tl.constexpr,
             x0_len: tl.constexpr, x1_len: tl.constexpr, x2_len: tl.constexpr, x3_len: tl.constexpr):
    
    total_dim1_len = x0_len + x1_len + x2_len + x3_len
    x0 = tl.load(x0_ptr + tl.arange(0, dim0_len * x0_len))
    x0 = x0.reshape(dim0_len, x0_len)
    x1 = tl.load(x1_ptr + tl.arange(0, dim0_len * x1_len))
    x1 = x1.reshape(dim0_len, x1_len)
    x2 = tl.load(x2_ptr + tl.arange(0, dim0_len * x2_len))
    x2 = x2.reshape(dim0_len, x2_len)
    x3 = tl.load(x3_ptr + tl.arange(0, dim0_len * x3_len))
    x3 = x3.reshape(dim0_len, x3_len)
    #  (86,5), (86,36), (86,5) (86,4)
    #torch.arange(0,86)[:,None] * 50 + torch.arange(0,36)
    idx_start = 0
    nidx0 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x0_len)
    tl.store(output_ptr + nidx0, x0)

    idx_start += x0_len
    nidx1 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x1_len)
    tl.store(output_ptr + nidx1, x1)

    idx_start += x1_len
    nidx2 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x2_len)
    tl.store(output_ptr + nidx2, x2)

    idx_start += x2_len
    nidx3 = (tl.arange(0, dim0_len)[:, None] * total_dim1_len + idx_start) + tl.arange(0, x3_len)
    tl.store(output_ptr + nidx3, x3)


def cat_dim0(data_type):
    x1_shape = (86, 48)
    x2_shape = (8, 48)
    x3_shape = (16, 48)

    x1 = torch.rand(x1_shape, dtype=data_type).npu()
    x2 = torch.rand(x2_shape, dtype=data_type).npu()
    x3 = torch.rand(x3_shape, dtype=data_type).npu()
    res = torch.zeros((x1_shape[0] + x2_shape[0] + x3_shape[0], 48), dtype=data_type).npu()
    fn3_dim0[(1,1,1)](res, x1, x2, x3, x1_shape[0]*x1_shape[1], x2_shape[0]*x2_shape[1], x3_shape[0]*x3_shape[1])

    res_ref = torch.cat((x1, x2, x3), dim=0)
    assert torch.allclose(res_ref, res,  rtol=1e-03, atol=1e-03, equal_nan=True)


def cat_dim1(data_type):
    #data_type = torch.float16
    x0_shape = (86,5)
    x1_shape = (86,36)
    x2_shape = (86,5)
    x3_shape = (86,4)

    dim = 1
    x0 = torch.rand(x0_shape, dtype=data_type).npu()
    x1 = torch.rand(x1_shape, dtype=data_type).npu()
    x2 = torch.rand(x2_shape, dtype=data_type).npu()
    x3 = torch.rand(x3_shape, dtype=data_type).npu()
    res = torch.zeros((86, x0_shape[dim] + x1_shape[dim] + x2_shape[dim] + x3_shape[dim]), dtype=data_type).npu()
    fn4_dim1[(1,1,1)](res, x0, x1, x2, x3, 86, x0_shape[1], x1_shape[1], x2_shape[1], x3_shape[1])
    #print("res_tri=", res)

    res_ref = torch.cat((x0, x1, x2, x3), dim = 1)
    #print("res_ref=", res_ref)
    assert torch.allclose(res_ref, res,  rtol=1e-03, atol=1e-03, equal_nan=True)

def test_cat():
    cat_dim0(torch.float16)
    cat_dim1(torch.float16)

