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

import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common
import os


os.environ["TRITON_DEVICE_PRINT"] = "1"
os.environ["TRITON_ENABLE_TASKQUEUE"] = "0"
shape = (8,)
XS = 8
XVALS_INT = [0,
            torch.iinfo(torch.int8).min,
            torch.iinfo(torch.int8).max,
            torch.iinfo(torch.int16).min,
            torch.iinfo(torch.int16).max,
            torch.iinfo(torch.int32).min,
            torch.iinfo(torch.int32).max,
            torch.iinfo(torch.int32).max+1]
XVALS_FP = [0,
            torch.finfo(torch.float32).eps,
            torch.finfo(torch.float16).eps,
            torch.finfo(torch.bfloat16).eps,
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float16).max,
            torch.finfo(torch.bfloat16).max,
            1]


def torch_func(x0, x1):
    res = x0 + x1
    return res


@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, XBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.device_print("OUTPUT = ", tmp2)
    tl.store(out_ptr0 + idx, tmp2)


def triton_func(x0, x1, XS):
    out = torch.empty_like(x0)
    triton_kernel[1, 1, 1](out, x0, x1, XS)
    return out


@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize('sigtype', ['int64'])
def test_device_print_int64(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.parametrize('sigtype', ['int32'])
def test_device_print_int32(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.parametrize('sigtype', ['int16'])
def test_device_print_int16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.parametrize('sigtype', ['int8'])
def test_device_print_int8(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.parametrize('sigtype', ['float32'])
def test_device_print_fp32(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.parametrize('sigtype', ['float16'])
def test_device_print_fp16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for bishengir-compile to support")
@pytest.mark.parametrize('sigtype', ['bfloat16'])
def test_device_print_bf16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)
