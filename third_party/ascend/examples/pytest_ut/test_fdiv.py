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


def torch_fdiv(x0, x1):
    res = x0 / x1
    return res


@triton.jit
def triton_fdiv(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        tmp0 = tl.load(in_ptr0 + x_index)
        tmp1 = tl.load(in_ptr1 + x_index)
        tmp2 = tl.fdiv(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                             ['float16', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_fdiv(param_list):
    # 生成数据
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_tmp = test_common.generate_tensor(shape, dtype)
    y0 = y_tmp.masked_fill(y_tmp == 0, 1)
    y0 = y0.npu()

    # torch结果
    y_ref = torch_fdiv(x0, y0).to(eval('torch.' + dtype))
    # triton结果
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_fdiv[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # 比较结果
    test_common.validate_cmp(dtype, y_cal, y_ref)
