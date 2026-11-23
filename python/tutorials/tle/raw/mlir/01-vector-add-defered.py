import types
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, llvm, nvvm, scf
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def edsl_core(
    output: Input[L["!llvm.ptr<1>"]],
    x: Input[L["!llvm.ptr<1>"]],
    y: Input[L["!llvm.ptr<1>"]],
    n_elements: Input[L["i32"]],
):
    tidx = nvvm.read_ptx_sreg_tid_x(ir.IntegerType.get_signless(32))
    bdimx = nvvm.read_ptx_sreg_ntid_x(ir.IntegerType.get_signless(32))
    gdimx = nvvm.read_ptx_sreg_nctaid_x(ir.IntegerType.get_signless(32))
    bidx = nvvm.read_ptx_sreg_ctaid_x(ir.IntegerType.get_signless(32))
    tidx = arith.index_cast(ir.IndexType.get(), tidx)
    bdimx = arith.index_cast(ir.IndexType.get(), bdimx)
    gdimx = arith.index_cast(ir.IndexType.get(), gdimx)
    bidx = arith.index_cast(ir.IndexType.get(), bidx)
    idx = arith.addi(arith.muli(bidx, bdimx), tidx)
    step = arith.muli(bdimx, gdimx)
    n_elements = arith.index_cast(ir.IndexType.get(), n_elements)
    for i in scf.for_(idx, n_elements, step):
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


def _bind_mlir_edsl(name: str, *, deferred: bool):
    fn = types.FunctionType(
        edsl_core.__code__,
        edsl_core.__globals__,
        name,
        edsl_core.__defaults__,
        edsl_core.__closure__,
    )
    return dialect(name="mlir", deferred=deferred, extern_func_name=name)(fn)


edsl_eager = _bind_mlir_edsl("edsl_eager", deferred=False)
edsl_deferred = _bind_mlir_edsl("edsl_deferred", deferred=True)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    DEFERRED: tl.constexpr,
):
    if DEFERRED:
        tle_raw.call(
            edsl_deferred,
            [output_ptr, x_ptr, y_ptr, n_elements],
            output_indices=[0],
        )
    else:
        tle_raw.call(edsl_eager, [output_ptr, x_ptr, y_ptr, n_elements])


def add(x: torch.Tensor, y: torch.Tensor, *, deferred: bool = False):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024, DEFERRED=deferred)
    return output


def run_case(*, deferred: bool) -> None:
    mode = "deferred" if deferred else "eager"
    edsl = edsl_deferred if deferred else edsl_eager
    print(f"--- tle_raw mode={mode} ---")
    print(f"edsl.deferred={edsl.deferred}")
    x = torch.randn(2048, device=DEVICE)
    y = torch.randn(2048, device=DEVICE)
    z = add(x, y, deferred=deferred)
    assert torch.allclose(x + y, z), (x + y, z)
    print(f"{mode}: OK")


if __name__ == "__main__":
    for deferred in (False, True):
        run_case(deferred=deferred)
