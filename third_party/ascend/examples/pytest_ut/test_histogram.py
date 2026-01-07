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
import pytest
import math
import test_common

@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [2])
@pytest.mark.parametrize("dtype", ["int32","int64"])
def test_histogram(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    print(y_cal)
    print(y_ref)    
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [2])
@pytest.mark.parametrize("dtype", [ "uint32", "uint64"])
def test_histogram_uint(M, N, ncore, dtype):
    torch.manual_seed(17)
    x_cpu = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}'), device="cpu")
    x = x_cpu.to("npu")
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.{dtype}'))
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)
