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
import functools
import os
import re

shape = (8,)
XS = 8
XVALS_INT = [0, 
             -128, # torch.iinfo(torch.int8).min
             127, # torch.iinfo(torch.int8).max
             -32768, # torch.iinfo(torch.int16).min
             32767, # torch.iinfo(torch.int16).max
             -2147483648, # torch.iinfo(torch.int32).min
             2147483647, # torch.iinfo(torch.int32).max
             9223372036854775807] # torch.iinfo(torch.int64).max
             
XVALS_FP = [0.0000000000e+00, # 0
            1.1921000009e-07, # torch.finfo(torch.float32).eps
            9.7655999707e-04, # torch.finfo(torch.float16).eps
            7.8125000000e-03, # torch.finfo(torch.bfloat16).eps
            3.4027999388e+38, # torch.finfo(torch.float32).max
            6.5504000000e+04, # torch.finfo(torch.float16).max
            3.3894999515e+38, # torch.finfo(torch.bfloat16).max
            1.0000000000e+00] # 1


def torch_func(x0, x1):
    res = x0 + x1
    return res


@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, XBLOCK: tl.constexpr, print_data_ptr: tl.constexpr,
                  assert_data_ptr: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.static_print(print_data_ptr)
    tl.static_assert(assert_data_ptr == assert_data_ptr, "assert_data should equal assert_data")
    tl.store(out_ptr0 + idx, tmp2)


def triton_func(x0, x1, XS, print_data_ptr, assert_data_ptr):
    out = torch.empty_like(x0)
    triton_kernel[1, 1, 1](out, x0, x1, XS, print_data_ptr, assert_data_ptr)
    return out


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['int8'])
@test_common.capture_output("-128")
def test_static_print_int8(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, -128, XVALS_INT[0])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['int16'])
@test_common.capture_output("-32768")
def test_static_print_int16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, -32768, XVALS_INT[2])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['int32'])
@test_common.capture_output("-2147483648")
def test_static_print_int32(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, -2147483648, XVALS_INT[4])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['int64'])
@test_common.capture_output("9223372036854775807")
def test_static_print_int64(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, 9223372036854775807, XVALS_INT[-1])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['float16'])
@test_common.capture_output("1.1921000009e-07")
def test_static_print_float16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, 1.1921000009e-07, XVALS_FP[1])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['float32'])
@test_common.capture_output("0.0078125")
def test_static_print_float32(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, 7.8125000000e-03, XVALS_FP[0])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['bfloat16'])
@test_common.capture_output("0.00097655999707")
def test_static_print_bfloat16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, 9.7655999707e-04, XVALS_FP[2])
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)


@pytest.mark.skip(reason="waiting for TA to support")
@pytest.mark.parametrize('sigtype', ['int8'])
@test_common.capture_output("True")
def test_static_print_bool(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype=dtype).npu()
    x1 = torch.ones(shape, dtype=dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS, True, True)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)