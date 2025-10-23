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
@test_common.capture_output("???")
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
@test_common.capture_output("0,-128,127,-32768,32767,-2147483648,2147483647,-2147483648")
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
@test_common.capture_output("0,-128,127,-32768,32767,0,-1,0")
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
@test_common.capture_output("0,-128,127,0,-1,0,-1,0")
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
@test_common.capture_output("0,1.19209e-07,0.000976562,0.0078125,3.40282e+38,65504,3.38953e+38,1")
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
@test_common.capture_output("0,1.19209e-07,0.000976562,0.0078125,inf,65504,inf,1")
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
@test_common.capture_output("???")
def test_device_print_bf16(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_FP[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)