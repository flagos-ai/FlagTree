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
import test_common


def torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query):
    idx = query[index]
    offs_m = torch.arange(offs_num, dtype=torch.int32).npu()
    query_pos = offs_m // divnum - idx
    mask = query_pos < maxindex

    tensor0 = torch.tensor(0, dtype=torch.int32).npu()
    tensor1 = torch.tensor(1, dtype=torch.int32).npu()
    result = torch.where(mask, tensor1, tensor0).npu()
    return result


@triton.jit
def divsiop_select_analysis_kernel1(index, out_ptr, query, offs_num: tl.constexpr, divnum: tl.constexpr,
                                    maxindex: tl.constexpr):
    idx = tl.load(query + index)
    offs_m = tl.arange(0, offs_num)

    query_pos = offs_m // divnum - idx

    mask = query_pos < maxindex
    query_mask = tl.where(mask, 1, 0).to(tl.int1)
    tl.store(out_ptr + tl.arange(0, offs_num), query_mask)


@triton.jit
def divsiop_select_analysis_kernel2(index, out_ptr, query, offs_num: tl.constexpr, divnum: tl.constexpr,
                                    maxindex: tl.constexpr):
    idx = tl.load(query + index)
    offs_m = tl.arange(0, offs_num)

    query_pos = -idx + offs_m // divnum
    mask = query_pos < maxindex

    query_mask = tl.where(mask, 1, 0).to(tl.int1)
    tl.store(out_ptr + tl.arange(0, offs_num), query_mask)


@pytest.mark.parametrize('param_list', [[16, 4, 2, index] for index in range(0, 4)])
def test_divsiop_select_analysis1(param_list):
    offs_num, divnum, maxindex, index = param_list
    query = torch.tensor(range(0, divnum)).npu()
    y_ref = torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    divsiop_select_analysis_kernel1[(1, )](index, y_cal, query, offs_num, divnum, maxindex)
    test_common.validate_cmp('int32', y_cal, y_ref)


@pytest.mark.parametrize('param_list', [[16, 4, 2, index] for index in range(0, 4)])
def test_divsiop_select_analysis2(param_list):
    offs_num, divnum, maxindex, index = param_list
    query = torch.tensor(range(0, divnum)).npu()
    y_ref = torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    divsiop_select_analysis_kernel2[(1, )](index, y_cal, query, offs_num, divnum, maxindex)
    test_common.validate_cmp('int32', y_cal, y_ref)
