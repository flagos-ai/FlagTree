import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

types = [
    "float32",
    "float16",
    "bfloat16",
]

shapes = [
    # 3,
    # 32,
    37,
    # 256,
    # 781,
]

@pytest.mark.parametrize("sigtype", types)
@pytest.mark.parametrize("N", shapes)
def test_isnan(sigtype, N):

    def torch_func(x0):
        res = torch.isnan(x0)
        return res


    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = tl.extra.ascend.libdevice.isnan(x0)
        tl.store(out_ptr0 + idx, ret)


    def triton_func(x0, N):
        out = torch.zeros(x0.shape, dtype=torch.bool).npu()
        triton_kernel[1, 1, 1](out, x0, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x0[1] = float('nan')

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0, N)
    test_common.validate_cmp("bool", triton_cal, torch_ref)
    assert triton_cal[1] == True

invalid_types = [
    "int32",
]

@pytest.mark.parametrize("sigtype", invalid_types)
@pytest.mark.parametrize("N", shapes)
@test_common.raises_with_match(triton.compiler.errors.CompilationError, "input arg type does not match.")
def test_isnan_invalid_dtype(sigtype, N):

    def torch_func(x0):
        res = torch.isnan(x0)
        return res


    @triton.jit
    def triton_kernel(out_ptr0, in_ptr0, N: tl.constexpr):
        idx = tl.arange(0, N)
        x0 = tl.load(in_ptr0 + idx)
        ret = tl.extra.ascend.libdevice.isnan(x0)
        tl.store(out_ptr0 + idx, ret)


    def triton_func(x0, N):
        out = torch.zeros(x0.shape, dtype=torch.bool).npu()
        triton_kernel[1, 1, 1](out, x0, N)
        return out

    x0 = test_common.generate_tensor(shape=(N,), dtype=sigtype).npu()
    x0[1] = float('nan')

    torch_ref = torch_func(x0)
    triton_cal = triton_func(x0, N)
    test_common.validate_cmp("bool", triton_cal, torch_ref)
    assert triton_cal[1] == True
