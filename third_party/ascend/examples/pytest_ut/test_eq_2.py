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


def torch_eq(x0, x1):
    return x0 == x1

def torch_eq_from_np(x0, x1, dtype):
    return torch.from_numpy(x0 == x1).to(eval('torch.' + dtype))

@triton.jit
def triton_eq(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index, None)
        tmp1 = tl.load(in_ptr1 + x_index, None)
        tmp2 = tmp0 == tmp1
        tl.store(out_ptr0 + x_index, tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['float16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['bfloat16', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int64', (2, 4096, 8), 2, 32768, 1024],
                             ['uint8', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint64', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_eq(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    np_x0 = test_common.generate_numpy(shape, dtype)
    x0 = torch.from_numpy(np_x0).to(eval('torch.' + dtype)).npu()
    np_x1 = test_common.generate_numpy(shape, dtype)
    x1 = torch.from_numpy(np_x1).to(eval('torch.' + dtype)).npu()
    # torch结果
    torch_res = torch_eq_from_np(np_x0, np_x1, dtype).npu()
    # triton结果
    triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_eq[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
