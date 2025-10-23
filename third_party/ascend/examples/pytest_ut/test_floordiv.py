import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


def torch_func(x0, x1):
    res = x0 // x1
    return res


@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, N: tl.constexpr):
    idx = tl.arange(0, N)
    x = tl.load(in_ptr0 + idx)
    y = tl.load(in_ptr1 + idx)
    ret = x // y
    tl.store(out_ptr0 + idx, ret)


def triton_func(x0, x1, N):
    out = torch.empty_like(x0)
    triton_kernel[1, 1, 1](out, x0, x1, N)
    return out


types = [
    "int32",
]

shapes = [
    3,
    32,
    37,
    256,
    781,
]

@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_floordiv(sigtype, N):
    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x1 = x1.masked_fill(x1 == 0, 1)

    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1, N)
    test_common.validate_cmp(sigtype, triton_cal, torch_ref)

