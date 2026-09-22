"""
TLE-Raw TOPS Backend: Vector Addition
======================================

This tutorial demonstrates how to use TLE-Raw with the TOPS C++ backend
to compile and run a simple vector addition kernel on GCU hardware.

The .tops file contains TOPS C++ device code that is compiled via topscc
to LLVM IR, then embedded into the Triton MLIR pipeline via TLE-Raw's
dsl_region mechanism.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="tops", file=Path(__file__).parent / "01-vector-add-tensor-dsl.tops", extern_func_name="VectorAdd",
         deferred=True)
def edsl_deferred(*args, **kwargs):
    ...


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tle_raw.call(edsl_deferred, [output_ptr + block_start, x, y], output_indices=[0])


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=32768)  # type: ignore[arg-type]
    return output


if __name__ == "__main__":
    x = torch.randn(32768 * 24, device=DEVICE)
    y = torch.randn(32768 * 24, device=DEVICE)
    z = add(x, y)
    assert torch.allclose(x + y, z), (x + y, z)
    print("TOPS Vector Add: PASSED")
