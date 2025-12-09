from typing import Annotated

from mlir import ir
from mlir.dialects import arith, nvvm, tensor
import torch
import triton
import triton.language as tl
from triton.experimental.flagtree.edsl import dialect
from triton.experimental import flagtree
import triton.experimental.flagtree.language as fl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@dialect(name="mlir")
def edsl(x: Annotated[ir.RankedTensorType, "tensor<1024xf32>"], y: Annotated[ir.RankedTensorType, "tensor<1024xf32>"],
         output: Annotated[ir.RankedTensorType, "tensor<1024xf32>"]):
    tidx = nvvm.ThreadIdXOp(ir.IntegerType.get_signless(32)).res
    bidx = nvvm.BlockIdXOp(ir.IntegerType.get_signless(32)).res
    bdimx = nvvm.BlockDimXOp(ir.IntegerType.get_signless(32)).res
    idx = arith.addi(arith.muli(bidx, bdimx), tidx)
    idx = arith.index_cast(ir.IndexType.get(), idx)
    xval = tensor.extract(x, [idx])
    yval = tensor.extract(y, [idx])
    result = arith.addf(xval, yval)
    tensor.insert(result, output, [idx])


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
    fl.call(edsl, [x, y, output])
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, num_warps=32)
    return output


if __name__ == "__main__":
    x = torch.randn(1024, device=DEVICE)
    y = torch.randn(1024, device=DEVICE)
    z = add(x, y)
    assert torch.allclose(x + y, z)
