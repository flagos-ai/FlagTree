from mlir import ir
from mlir.dialects import arith, llvm, nvvm, scf
import torch
import triton
import triton.language as tl
from triton.experimental import flagtree
from triton.experimental.flagtree.edsl import dialect, Input
import triton.experimental.flagtree.language as fl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="mlir")
def edsl(output: Input["!llvm.ptr<1>"], x: Input["!llvm.ptr<1>"], y: Input["!llvm.ptr<1>"],  # noqa: F722,
         n_elements: Input["i32"]):  # noqa: F821
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    tidx = arith.index_cast(ir.IndexType.get(), tidx)
    bdimx = arith.index_cast(ir.IndexType.get(), bdimx)
    bidx = arith.index_cast(ir.IndexType.get(), bidx)
    idx = arith.addi(arith.muli(bidx, bdimx), tidx)
    n_elements = arith.index_cast(ir.IndexType.get(), n_elements)
    for i in scf.for_(idx, n_elements, bdimx):
        i = arith.index_cast(ir.IntegerType.get_signless(32), i)
        ptrty = ir.Type.parse("!llvm.ptr<1>")
        f32ty = ir.Type.parse("f32")
        xptr = llvm.getelementptr(ptrty, x, [i], [-2147483648], f32ty, 0)
        yptr = llvm.getelementptr(ptrty, y, [i], [-2147483648], f32ty, 0)
        xval = llvm.load(f32ty, xptr)
        yval = llvm.load(f32ty, yptr)
        outval = arith.addf(xval, yval)
        outptr = llvm.getelementptr(ptrty, output, [i], [-2147483648], f32ty, 0)
        llvm.store(outval, outptr)
        scf.yield_([])


@flagtree.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    fl.call(edsl, [], [output_ptr, x_ptr, y_ptr, n_elements])


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
