"""
TLE-Raw GCU MLIR EDSL: Vector Addition (Scalar per-element)
============================================================

Uses @dialect(name="tops_mlir") with default use_gcu_opt=True.
Lowering is fully handled by gcu-compiler-opt:
  - !gcu.ptr<f32> parameters preserved natively
  - gcu.ptr2memref converts !gcu.ptr<f32> → memref<?xf32>
  - Standard memref.load / memref.store for element access
  - gpu.thread_id / gpu.block_id for hardware indexing
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu, memref, scf
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="tops_mlir")
def edsl(
    output: Input[L["!gcu.ptr<f16>"]],
    x: Input[L["!gcu.ptr<f16>"]],
    y: Input[L["!gcu.ptr<f16>"]],
    n_elements: Input[L["i32"]],
):
    idx_ty = ir.IndexType.get()
    f16 = ir.F16Type.get()
    dyn = ir.ShapedType.get_dynamic_size()
    memref_f16 = ir.MemRefType.get([dyn], f16)

    mr_x = gcu.ptr2memref(x, memref_f16)
    mr_y = gcu.ptr2memref(y, memref_f16)
    mr_out = gcu.ptr2memref(output, memref_f16)

    tidx = gpu.thread_id(gpu.Dimension.x)
    bdimx = gpu.block_dim(gpu.Dimension.x)
    gdimx = gpu.grid_dim(gpu.Dimension.x)
    bidx = gpu.block_id(gpu.Dimension.x)
    idx = arith.addi(arith.muli(bidx, bdimx), tidx)
    step = arith.muli(bdimx, gdimx)
    n_elements = arith.index_cast(idx_ty, n_elements)
    for i in scf.for_(idx, n_elements, step):
        xval = memref.load(mr_x, [i])
        yval = memref.load(mr_y, [i])
        outval = arith.addf(xval, yval)
        memref.store(outval, mr_out, [i])
        scf.yield_([])


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    tle_raw.call(edsl, [output_ptr, x_ptr, y_ptr, n_elements])


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    y = torch.randn(2048, device=DEVICE, dtype=torch.float16)
    z = add(x, y)
    assert torch.allclose(x + y, z), (x + y, z)
    print("PASSED")
