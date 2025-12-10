from typing import Annotated

from mlir import ir
from mlir.dialects import arith, math, memref, nvvm, scf
import torch
import triton
import triton.language as tl
from triton.experimental import flagtree
from triton.experimental.flagtree.edsl import dialect
import triton.experimental.flagtree.language as fl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def naive_softmax(x):
    x_max, _ = x.max(dim=1)
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


@dialect(name="mlir")
def edsl(y: Annotated[ir.MemRefType, "memref<?xf32>"], x: Annotated[ir.MemRefType, "memref<?xf32>"]):
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    tidx = arith.index_cast(ir.IndexType.get(), tidx)
    bdimx = arith.index_cast(ir.IndexType.get(), bdimx)
    length = memref.dim(y, arith.constant(ir.IndexType.get(), 0))
    for i in scf.for_(tidx, length, bdimx):
        val = memref.load(x, [i])
        memref.store(val, y, [i])
        scf.yield_([])
    whileop = scf.while_(
        [ir.IndexType.get()],
        [length],
    )
    condblock = whileop.owner.opview.before.blocks.append(ir.IndexType.get())
    doblock = whileop.owner.opview.after.blocks.append(ir.IndexType.get())
    with ir.InsertionPoint(condblock):
        [arg] = condblock.arguments
        cond = arith.cmpi(arith.CmpIPredicate.sgt, arg, arith.constant(ir.IndexType.get(), 1))
        scf.condition(cond, [arg])
    with ir.InsertionPoint(doblock):
        [arg] = doblock.arguments
        half = arith.ceildivsi(arg, arith.constant(ir.IndexType.get(), 2))
        for i in scf.for_(tidx, half, bdimx):
            ok = arith.cmpi(arith.CmpIPredicate.slt, i, half)
            ifop = scf.if_([], ok)
            then = ifop.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(then):
                left = tidx
                right = arith.addi(tidx, half)
                left_val = memref.load(y, [left])
                right_val = memref.load(y, [right])
                max_val = arith.maxnumf(left_val, right_val)
                memref.store(max_val, y, [left])
                scf.yield_([])
            scf.yield_([])
        nvvm.barrier0()
        scf.yield_([half])
    max_val = memref.load(y, [arith.constant(ir.IndexType.get(), 0)])
    for i in scf.for_(tidx, length, bdimx):
        val = memref.load(x, [i])
        val = arith.subf(val, max_val)
        val = math.exp(val)
        memref.store(val, y, [i])
        scf.yield_([])
    whileop = scf.while_(
        [ir.IndexType.get()],
        [length],
    )
    condblock = whileop.owner.opview.before.blocks.append(ir.IndexType.get())
    doblock = whileop.owner.opview.after.blocks.append(ir.IndexType.get())
    with ir.InsertionPoint(condblock):
        [init] = condblock.arguments
        cond = arith.cmpi(arith.CmpIPredicate.sgt, init, arith.constant(ir.IndexType.get(), 1))
        scf.condition(cond, [init])
    with ir.InsertionPoint(doblock):
        [init] = doblock.arguments
        half = arith.divsi(init, arith.constant(ir.IndexType.get(), 2))
        for i in scf.for_(tidx, half, bdimx):
            ok = arith.cmpi(arith.CmpIPredicate.slt, i, half)
            ifop = scf.if_([], ok)
            then = ifop.opview.thenRegion.blocks.append()
            with ir.InsertionPoint(then):
                left = tidx
                right = arith.addi(tidx, half)
                left_val = memref.load(y, [left])
                right_val = memref.load(y, [right])
                sum_val = arith.addf(left_val, right_val)
                memref.store(sum_val, y, [left])
                scf.yield_([])
            scf.yield_([])
        nvvm.barrier0()
        scf.yield_([half])
    sum_val = memref.load(y, [arith.constant(ir.IndexType.get(), 0)])
    for i in scf.for_(tidx, length, bdimx):
        val = memref.load(x, [i])
        val = arith.subf(val, max_val)
        val = math.exp(val)
        val = arith.divf(val, sum_val)
        memref.store(val, y, [i])
        scf.yield_([])


@flagtree.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        softmax_output = tl.zeros_like(row)
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        fl.call(edsl, [softmax_output, row])
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(8, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 4, device=DEVICE)
    y_triton = softmax(x)
    y_torch = naive_softmax(x)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
