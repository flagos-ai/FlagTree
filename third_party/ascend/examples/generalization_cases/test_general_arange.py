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

import math
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common
from test_common import TestUtils


def torch_pointwise(length):
    res = (torch.arange(0, length) / 2.7) * torch.arange(0, length)
    return res


def torch_arange(start, end):
    TRITON_MAX_TENSOR_NUMEL = 1048576
    if end < start:
        raise ValueError("arange's end argument must be greater than the start argument")
    if end - start > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")
    return torch.arange(start, end)


@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)


@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
def test_case(shape):
    start = 0
    end = shape[0]
    shape = [end - start]
    block = end - start
    dtype = 'int32'

    y_ref = torch_arange(start, end)
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()

    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)

    assert torch.equal(y_cal.cpu(), y_ref.cpu())