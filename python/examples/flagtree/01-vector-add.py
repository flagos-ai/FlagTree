from mlir import ir
from mlir.dialects import arith, memref, nvvm, scf
import torch
import triton
import triton.language as tl
from triton.experimental import flagtree
from triton.experimental.flagtree.edsl import dialect, Input, InOut
import triton.experimental.flagtree.language as fl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="mlir")
def edsl(output: InOut["?xf32"], x: Input["?xf32"], y: Input["?xf32"]):  # noqa: F722
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    tidx = arith.index_cast(ir.IndexType.get(), tidx)
    bdimx = arith.index_cast(ir.IndexType.get(), bdimx)
    length = memref.dim(output, arith.constant(ir.IndexType.get(), 0))
    for i in scf.for_(tidx, length, bdimx):
        xval = memref.load(x, [i])
        yval = memref.load(y, [i])
        result = arith.addf(xval, yval)
        memref.store(result, output, [i])
        scf.yield_([])


@flagtree.jit
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
    output = tl.zeros_like(x)
    output = fl.call(edsl, [output], [x, y])
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    x = torch.randn(2048, device=DEVICE)
    y = torch.randn(2048, device=DEVICE)
    z = add(x, y)
    assert torch.allclose(x + y, z), (x + y, z)
