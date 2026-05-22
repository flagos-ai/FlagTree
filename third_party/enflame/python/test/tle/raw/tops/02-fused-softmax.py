"""
TLE-Raw TOPS Backend: Fused Softmax
====================================

This tutorial demonstrates how to use TLE-Raw with the TOPS C++ backend
to compile and run a fused softmax kernel on GCU hardware.

The .tops file contains a softmax implementation using TOPS C++ device code,
compiled via topscc to LLVM IR.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="tops", file=Path(__file__).parent / "02-fused-softmax.tops")
def edsl(*args, **kwargs):
    ...


def naive_softmax(x):
    x_max, _ = x.max(dim=1)
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


@triton.jit
def softmax_kernel(output_ptr, input_ptr, n_rows, n_cols, input_row_stride, output_row_stride,
                   BLOCK_SIZE: tl.constexpr):
    tle_raw.call(edsl, [output_ptr, input_ptr, n_rows, n_cols, input_row_stride, output_row_stride])


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, 1, 1)](y, x, n_rows, n_cols, x.stride(0), y.stride(0), BLOCK_SIZE)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 4, device=DEVICE)
    y_triton = softmax(x)
    y_torch = naive_softmax(x)
    assert torch.allclose(y_triton, y_torch, atol=1e-4, rtol=1e-4), (y_triton, y_torch)
    print("TOPS Fused Softmax: PASSED")
