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
from triton.language.extra.ascend.libdevice import pow
import torch
import torch_npu
import pytest
import test_common

types = [
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int16",
    "int8",
]

shapes = [
    # 3,
    # 32,
    37,
    # 256,
    # 781,
]

@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_pow_vv(sigtype, N):

    def torch_func(x0, x1):
        res = torch.pow(x0, x1)
        return res

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        x1 = tl.load(in_ptr1 + idx)
        ret = pow(x0, x1)
        tl.store(out_ptr0 + idx, ret)

    def triton_func(x0, x1, N):
        out = torch.empty_like(x0)
        triton_kernel[1, 1, 1](out, x0, x1, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()

    triton_cal = triton_func(x0, x1, N)
    torch_ref = torch_func(x0, x1)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)

@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_pow_vs_dynamic(sigtype, N):

    def torch_func(x0, x1):
        res = torch.pow(x0, x1)
        return res

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        x1 = tl.load(in_ptr1)
        ret = pow(x0, x1)
        tl.store(out_ptr0 + idx, ret)

    def triton_func(x0, x1, N):
        out = torch.empty_like(x0)
        triton_kernel[1, 1, 1](out, x0, x1, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(1,), dtype=sigtype).npu()

    triton_cal = triton_func(x0, x1, N)
    torch_ref = torch_func(x0, x1)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)

@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_pow_vs_const(sigtype, N):

    def torch_func(x0, x1):
        res = torch.pow(x0, x1)
        return res

    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, x1: tl.constexpr, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = pow(x0, x1)
        tl.store(out_ptr0 + idx, ret)

    def triton_func(x0, x1, N):
        out = torch.empty_like(x0)
        triton_kernel[1, 1, 1](out, x0, x1, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(1,), dtype=sigtype).item()

    triton_cal = triton_func(x0, x1, N)
    torch_ref = torch_func(x0, x1)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)