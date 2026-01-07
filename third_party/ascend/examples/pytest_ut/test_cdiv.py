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
import torch
import torch_npu
import test_common


def torch_cdiv(x0, x1, dtype_x):
    if dtype_x in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.div(x0, x1, rounding_mode='floor') + (x0 % x1 != 0).to(torch.int)
    else:
        if dtype_x in ["float16", "bfloat16"]:
            x0 = x0.to(torch.float32)
            x1 = x1.to(torch.float32)
        return torch.ceil(x0 / x1).to(eval("torch." + dtype_x))
    

def torch_cdiv_special(x0, x1, dtype_x):
    if dtype_x in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        return torch.div(x0, x1, rounding_mode='floor') + (x0 % x1 != 0).to(torch.int)
    else:
        if dtype_x in ["float16", "bfloat16"]:
            x0 = x0.to(torch.float32)
        return torch.ceil(x0 / x1).to(eval("torch." + dtype_x))


@triton.jit
def triton_cdiv(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = tl.cdiv(XBLOCK, XBLOCK_SUB)
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tl.cdiv(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, None)


@triton.jit
def triton_cdiv_special(in_ptr0, div, out_ptr0, XBLOCK, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = tl.cdiv(XBLOCK, XBLOCK_SUB)
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp2 = tl.cdiv(tmp0, div)
        tl.store(out_ptr0 + x_index, tmp2, None)


param_lists = [
    # ['int8', (4096,), 1, 4096, 4096],
    # ['int16', (4096,), 1, 4096, 4096],
    ['int32', (4096,), 1, 4096, 4096],
    # ['int64', (4096,), 1, 4096, 4096],
    # ['float16', (4096,), 1, 4096, 4096],
    ['float32', (4096,), 1, 4096, 4096],
    # ['bfloat16', (4096,), 1, 4096, 4096],
]


@pytest.mark.parametrize('param_list', param_lists)
def test_cdiv(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    torch_dtype = eval('torch.' + dtype)
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    torch_res = torch_cdiv(x0, x1, dtype)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_cdiv[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)


@pytest.mark.parametrize('param_list', param_lists)
def test_cdiv_special(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    torch_dtype = eval('torch.' + dtype)
    x0 = test_common.generate_tensor(shape, dtype).npu()
    x1 = 2
    # torch结果
    torch_res = torch_cdiv_special(x0, x1, dtype)
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_cdiv_special[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
