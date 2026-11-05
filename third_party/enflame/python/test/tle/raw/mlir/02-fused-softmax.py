"""
TLE-Raw GCU MLIR EDSL: Fused Softmax (Scalar per-element)
==========================================================

Row-wise softmax using scalar memref.load / memref.store.

Algorithm (3-pass):
  Pass 1: Find row-wise max
  Pass 2: Compute exp(x - max) and accumulate sum
  Pass 3: Normalize: result = exp_val / sum

Grid: (n_rows,), each block handles one row via thread 0.

Architecture: GCU400
Lowering: gcu-compiler-opt (use_gcu_opt=True, default)
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu, math as math_d, memref, scf
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
NEG_INF = float('-inf')


@dialect(name="tops_mlir", use_gcu_opt=True)
def softmax_edsl(
    output: Input[L["!gcu.ptr<f32>"]],
    input_ptr: Input[L["!gcu.ptr<f32>"]],
    input_row_stride: Input[L["i32"]],
    output_row_stride: Input[L["i32"]],
    n_cols: Input[L["i32"]],
):
    idx_ty = ir.IndexType.get()
    i32 = ir.IntegerType.get_signless(32)
    f32 = ir.F32Type.get()
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_f32 = ir.MemRefType.get([dyn], f32)

    c0_i32 = arith.constant(i32, ir.IntegerAttr.get(i32, 0))
    c1_idx = arith.constant(idx_ty, ir.IntegerAttr.get(idx_ty, 1))
    neg_inf = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
    zero_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))

    in_mr = gcu.ptr2memref(input_ptr, memref_dyn_f32)
    out_mr = gcu.ptr2memref(output, memref_dyn_f32)

    row_idx = gpu.block_id(gpu.Dimension.x)
    row_idx_i32 = arith.index_cast(i32, row_idx)
    tid = gpu.thread_id(gpu.Dimension.x)
    tid_i32 = arith.index_cast(i32, tid)

    in_row_off = arith.muli(row_idx_i32, input_row_stride)
    out_row_off = arith.muli(row_idx_i32, output_row_stride)

    n_cols_idx = arith.index_cast(idx_ty, n_cols)
    in_base_idx = arith.index_cast(idx_ty, in_row_off)
    out_base_idx = arith.index_cast(idx_ty, out_row_off)
    c0_idx = arith.constant(idx_ty, ir.IntegerAttr.get(idx_ty, 0))

    is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid_i32, c0_i32)
    if_t0 = scf.IfOp(is_t0)
    with ir.InsertionPoint(if_t0.then_block):
        # === Pass 1: Find row max ===
        pass1 = scf.ForOp(c0_idx, n_cols_idx, c1_idx, [neg_inf])
        with ir.InsertionPoint(pass1.body):
            prev_max = pass1.inner_iter_args[0]
            col_idx = arith.addi(in_base_idx, pass1.induction_variable)
            val = memref.load(in_mr, [col_idx])
            new_max = arith.maximumf(prev_max, val)
            scf.yield_([new_max])

        row_max = pass1.results[0]

        # === Pass 2: exp(x - max) + sum ===
        pass2 = scf.ForOp(c0_idx, n_cols_idx, c1_idx, [zero_f32])
        with ir.InsertionPoint(pass2.body):
            prev_sum = pass2.inner_iter_args[0]
            in_col = arith.addi(in_base_idx, pass2.induction_variable)
            out_col = arith.addi(out_base_idx, pass2.induction_variable)
            val = memref.load(in_mr, [in_col])
            diff = arith.subf(val, row_max)
            exp_val = math_d.exp(diff)
            memref.store(exp_val, out_mr, [out_col])
            new_sum = arith.addf(prev_sum, exp_val)
            scf.yield_([new_sum])

        row_sum = pass2.results[0]

        # === Pass 3: Normalize ===
        pass3 = scf.ForOp(c0_idx, n_cols_idx, c1_idx)
        with ir.InsertionPoint(pass3.body):
            out_col = arith.addi(out_base_idx, pass3.induction_variable)
            exp_val = memref.load(out_mr, [out_col])
            result = arith.divf(exp_val, row_sum)
            memref.store(result, out_mr, [out_col])
            scf.yield_([])

        scf.yield_([])


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols):
    tle_raw.call(softmax_edsl, [output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols])


def softmax(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(n_rows, )](y, x, x.stride(0), y.stride(0), n_cols, num_warps=4)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 256, device=DEVICE, dtype=torch.float32)
    y_edsl = softmax(x)
    y_ref = torch.softmax(x, dim=1)

    if torch.allclose(y_edsl, y_ref, atol=1e-5, rtol=1e-5):
        print("PASSED")
    else:
        max_diff = (y_edsl - y_ref).abs().max().item()
        print(f"FAILED: max_diff={max_diff:.6e}")
        print(f"  edsl[:4] = {y_edsl[0, :4].tolist()}")
        print(f"  ref[:4]  = {y_ref[0, :4].tolist()}")
