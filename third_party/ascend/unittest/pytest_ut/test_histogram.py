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
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)


@triton.jit
def histogram_mask_kernel(x_ptr, z_ptr, n_elements, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset1 = tl.arange(0, BLOCK_SIZE)
    offset2 = tl.arange(0, N)
    x_idx = block_start + offset1
    mask = x_idx < n_elements
    x = tl.load(x_ptr + x_idx)
    z = tl.histogram(x, N, mask=mask)
    tl.atomic_add(z_ptr + offset2, z)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_histogram(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}')).npu()
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
@pytest.mark.skipif(
    not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch),
    reason="uint32/uint64 are only supported on Ascend 950 (A5); A2/A3 load/histogram do not support them.",
)
def test_histogram_uint(M, N, ncore, dtype):
    torch.manual_seed(17)
    x_cpu = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}'), device="cpu")
    x = x_cpu.to("npu")
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.{dtype}'))
    # triton结果
    y_ref = torch.empty(N, dtype=eval(f'torch.{dtype}'), device="npu")
    histogram_kernel[(ncore, )](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@pytest.mark.skip(reason="not supported before the NPUIR is updated, and will be fixed later.")
@pytest.mark.parametrize("M", [2048])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("BLOCK_SIZE", [10])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_histogram_with_mask(M, N, BLOCK_SIZE, dtype):
    torch.manual_seed(42)

    x = torch.randint(low=0, high=N, size=(M, ), dtype=eval(f'torch.{dtype}')).npu()
    mask = torch.zeros((M, ), dtype=torch.bool)
    mask[:M // 2] = True

    valid_x = x.cpu()[mask]
    ref_count = torch.bincount(valid_x, minlength=N)
    y_cal = ref_count.to(eval(f'torch.{dtype}'))
    y_ref = torch.zeros(N, dtype=eval(f'torch.{dtype}'), device="npu")
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), )
    histogram_mask_kernel[grid](x, y_ref, n_elements=M // 2, N=N, BLOCK_SIZE=BLOCK_SIZE)
    test_common.validate_cmp(dtype, y_cal, y_ref)
