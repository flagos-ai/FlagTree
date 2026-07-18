"""
TLE-Raw TOPS Backend: Matrix Multiplication
=============================================

This tutorial demonstrates how to use TLE-Raw with the TOPS C++ backend
to compile and run a matrix multiplication kernel on GCU hardware.

The .tops file contains a simple GEMM implementation using TOPS C++ device
code, compiled via topscc to LLVM IR.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="tops", file=Path(__file__).parent / "03-matrix-multiplication.tops")
def edsl(*args, **kwargs):
    ...


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    tle_raw.call(edsl, [c_ptr, a_ptr, b_ptr, M, N, K])


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (1, )
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_SIZE=1)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.rand((64, 64), device=DEVICE, dtype=torch.float32) - 0.5
    b = torch.rand((64, 64), device=DEVICE, dtype=torch.float32) - 0.5
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp32_inputs={triton_output}")
    print(f"torch_output_with_fp32_inputs={torch_output}")

    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=1e-2)
    print("TOPS Matrix Multiplication: PASSED")
