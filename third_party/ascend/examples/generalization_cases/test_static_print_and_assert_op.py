import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common

import os

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["PYTEST_ADDOPTS"] = "-sv"

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

def torch_func(x0, x1):
    res = x0 + x1
    return res

@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, XBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.static_print(XBLOCK)
    tl.static_print(tmp2)
    tl.static_assert(XBLOCK == 8)
    tl.store(out_ptr0 + idx, tmp2)

def triton_func(x0, x1, XS):
    out = torch.empty_like(x0)
    triton_kernel[1,](out, x0, x1, XS)
    return out

@pytest.mark.parametrize('sigtype', ['int32', 'int64', 'int16', 'int8',
                                     'float32', 'float16','bfloat16'])
def test_static_print_and_assert(capsys, sigtype):
    dtype = eval(f"torch.{sigtype}")
    x0 = torch.zeros(shape, dtype = dtype).npu()
    x1 = torch.ones(shape, dtype = dtype).npu()
    for i in range(x1.numel()):
        x1[i] = XVALS_INT[i]
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, XS)
    captured = capsys.readouterr()

    if sigtype == "float32":
        assert "fp32" in captured.out
    if sigtype == "float16":
        assert "fp16" in captured.out
    if sigtype == "bfloat16":
        assert "bf16" in captured.out
    if "int" in sigtype:
        assert sigtype in captured.out
    assert "8" in captured.out

    test_common.validate_cmp(sigtype, triton_cal, torch_ref)