import textwrap

import numpy as np
import pytest
import torch

import triton
import triton.language as tl

import inspect
import benchmark

benchmark.select_cpu_backend()
from numpy.random import RandomState

from triton._internal_testing import (
    integral_dtypes,
    int_dtypes,
    str_to_triton_dtype,
    uint_dtypes,
    float_dtypes,
    float_dtypes_with_bfloat16,
    dtypes,
    dtypes_with_bfloat16,
    is_cuda,
    is_interpreter,
    is_hopper,
    is_hip,
    is_hip_cdna,
    is_hip_mi200,
    is_hip_mi300,
    is_hip_mi350,
    is_xpu,
    get_arch,
    torch_float8_dtypes,
    torch_dtypes,
    numpy_random,
    to_triton,
    torch_dtype_name,
    to_numpy,
)


def check_type_supported(dtype, device):
    '''
    skip test if dtype is not supported on the current device
    '''
    if device in ['cuda']:
        cc = torch.cuda.get_device_capability()
        if cc[0] < 8 and (dtype is tl.bfloat16 or dtype == "bfloat16" or dtype is torch.bfloat16):
            pytest.skip("bfloat16 is only supported on NVGPU with cc >= 80")
        if cc[0] < 9 and dtype in {tl.float8e4nv, "float8e4nv", "float8_e4m3fn"}:
            pytest.skip("float8e4nv is only supported on NVGPU with cc >= 90")
    if is_interpreter():
        if dtype in [tl.bfloat16, "bfloat16", torch.bfloat16]:
            pytest.skip("bfloat16 is not supported in the interpreter")


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype", [dtype for dtype in ["float32"]])
def test_math_erf_op(dtype, device):
    check_type_supported(dtype, device)
    SIZE = 128

    @triton.jit
    def kernel(Z, X, SIZE: tl.constexpr):
        off = tl.arange(0, SIZE)
        x = tl.load(X + off)
        z = tl.math.erf(x)
        tl.store(Z + off, z)

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    x = torch.randn(SIZE, dtype=torch_dtype, device=device)
    z_ref = torch.erf(x)
    z_tri = torch.zeros_like(x)
    kernel[(1, )](z_tri, x, SIZE=SIZE, num_warps=4)
    torch.testing.assert_close(z_tri, z_ref)


if __name__ == "__main__":
    test_math_erf_op("float32", "cpu")
