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
import numpy as np
import test_common


def torch_logical_and(x0, x1):
    res = torch.logical_and(x0, x1)
    return res


@triton.jit
def triton_logical_and(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    for loop1 in range(loops1):
        x_index = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + x_index)
        tmp1 = tl.load(in_ptr1 + x_index)
        tmp2 = tmp0.logical_and(tmp1)
        tl.store(out_ptr0 + x_index, tmp2)


@pytest.mark.parametrize('param_list',
                         [
                             ['bool', (2, 4096, 8), 2, 32768, 1024],
                             ['int8', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['int64', (2, 4096, 8), 2, 32768, 1024],
                             ['uint8', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['uint64', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                            #  ['float32', (2, 4096, 8), 2, 32768, 1024],
                            #  ['bfloat16', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_and(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    torch_dtype = eval('torch.' + dtype)
    np_x0 = test_common.generate_numpy(shape, dtype)
    np_x1 = test_common.generate_numpy(shape, dtype)
    x0 = torch.from_numpy(np_x0).to(torch_dtype).npu()
    x1 = torch.from_numpy(np_x1).to(torch_dtype).npu()
    # torch结果
    np_res = np.logical_and(np_x0, np_x1)
    torch_res = torch.from_numpy(np_res).to(torch.bool)
    # triton结果
    triton_res = torch.zeros(shape, dtype=torch.bool).npu()
    triton_logical_and[ncore, 1, 1](x0, x1, triton_res, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, triton_res, torch_res)
