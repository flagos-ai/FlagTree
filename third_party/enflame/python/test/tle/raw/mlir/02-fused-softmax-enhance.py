"""
TLE-Raw GCU MLIR EDSL: Fused Softmax (Hybrid Static/Dynamic)
=============================================================

Optimization strategy:
  - Static kernels cover block_size 256..1280:
    * 256/512/768/1024: 2-vector ILP (STEP=256), aggressive unrolling.
    * 1152/1280: single-vector (STEP=128), lower register pressure.
  - block_size alignment: ceil(n_cols/STEP)*STEP for <=1024, ceil(n_cols/VEC)*VEC for >1024.
  - Dynamic fallback (N > 1280): 2-pass ONLINE softmax.
    * Pass 1: online max+sum in a single read (correction: d *= exp(m_old - m_new)).
    * Pass 2: normalize exp(x-max)/sum + write output.
    * 3-tier per pass: 4-vec main + 1-vec middle + masked tail.
    * Memory traffic: 2R+1W (vs 3R+1W in 3-pass → 25% reduction).
  - Memory safety: offsets clamped to row boundaries + masking.

Architecture: GCU400 (4 SICs x 6 SIPs = 24 SIPs)
Lowering: gcu-compiler-opt (use_gcu_opt=True, default)
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu, math as math_d, memref as memref_d, scf
from mlir.dialects import vector as vector_d
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu
import triton.experimental.tle.language.raw as tle_raw

NEG_INF = float('-inf')
VEC = 128
STEP = 2 * VEC


def _ci32(i32, v):
    return arith.constant(i32, ir.IntegerAttr.get(i32, v))


def _cidx(idx, v):
    return arith.constant(idx, ir.IntegerAttr.get(idx, v))


def _build_3pass_body(in_memref, out_memref, vec_f32, vec_i32, vec_neg_inf, vec_zero, row_off_i32, bcast_n,
                      n_cols_m_vec, upper_idx, i32, f32, idx, c0_idx, c_step_idx):
    """Emit the 3-pass softmax logic using scf.ForOp with 2-vector ILP (STEP=256)."""
    c_vec_i32 = _ci32(i32, VEC)
    neg_inf_f32 = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
    one_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 1.0))

    # === Pass 1: find max ===
    loop1 = scf.ForOp(c0_idx, upper_idx, c_step_idx, [vec_neg_inf])
    with ir.InsertionPoint(loop1.body):
        acc = loop1.inner_iter_args[0]
        iv = arith.index_cast(i32, loop1.induction_variable)
        col_v0 = gcu.vector_step(iv, vec_i32)
        mask0 = arith.cmpi(arith.CmpIPredicate.slt, col_v0, bcast_n)
        safe_iv0 = arith.minsi(iv, n_cols_m_vec)
        abs_off0 = arith.addi(row_off_i32, safe_iv0)
        mem_idx0 = arith.index_cast(idx, abs_off0)
        v0 = gcu.maskedload(in_memref, mem_idx0, mask0, vec_neg_inf, vec_f32)
        iv1 = arith.addi(iv, c_vec_i32)
        col_v1 = gcu.vector_step(iv1, vec_i32)
        mask1 = arith.cmpi(arith.CmpIPredicate.slt, col_v1, bcast_n)
        safe_iv1 = arith.minsi(iv1, n_cols_m_vec)
        abs_off1 = arith.addi(row_off_i32, safe_iv1)
        mem_idx1 = arith.index_cast(idx, abs_off1)
        v1 = gcu.maskedload(in_memref, mem_idx1, mask1, vec_neg_inf, vec_f32)
        new_acc = arith.maximumf(arith.maximumf(acc, v0), v1)
        scf.yield_([new_acc])
    max_vec = loop1.results[0]
    scalar_max = vector_d.ReductionOp(f32, vector_d.CombiningKind.MAXIMUMF, max_vec).result
    bcast_max = gcu.vector_broadcast(scalar_max, vec_f32)

    # === Pass 2: accumulate sum = Σ exp(x-max) (read-only, no output writes) ===
    loop2 = scf.ForOp(c0_idx, upper_idx, c_step_idx, [vec_zero])
    with ir.InsertionPoint(loop2.body):
        acc = loop2.inner_iter_args[0]
        iv = arith.index_cast(i32, loop2.induction_variable)
        col_v0 = gcu.vector_step(iv, vec_i32)
        mask0 = arith.cmpi(arith.CmpIPredicate.slt, col_v0, bcast_n)
        safe_iv0 = arith.minsi(iv, n_cols_m_vec)
        abs_off0 = arith.addi(row_off_i32, safe_iv0)
        mem_idx0 = arith.index_cast(idx, abs_off0)
        v0 = gcu.maskedload(in_memref, mem_idx0, mask0, vec_neg_inf, vec_f32)
        exp_v0 = math_d.exp(arith.subf(v0, bcast_max))
        iv1 = arith.addi(iv, c_vec_i32)
        col_v1 = gcu.vector_step(iv1, vec_i32)
        mask1 = arith.cmpi(arith.CmpIPredicate.slt, col_v1, bcast_n)
        safe_iv1 = arith.minsi(iv1, n_cols_m_vec)
        abs_off1 = arith.addi(row_off_i32, safe_iv1)
        mem_idx1 = arith.index_cast(idx, abs_off1)
        v1 = gcu.maskedload(in_memref, mem_idx1, mask1, vec_neg_inf, vec_f32)
        exp_v1 = math_d.exp(arith.subf(v1, bcast_max))
        new_acc = arith.addf(arith.addf(acc, exp_v0), exp_v1)
        scf.yield_([new_acc])
    sum_vec = loop2.results[0]
    scalar_sum = vector_d.ReductionOp(f32, vector_d.CombiningKind.ADD, sum_vec).result
    inv_sum = arith.divf(one_f32, scalar_sum)
    bcast_inv = gcu.vector_broadcast(inv_sum, vec_f32)

    # === Pass 3: recompute exp(x-max)/sum and store to output ===
    loop3 = scf.ForOp(c0_idx, upper_idx, c_step_idx, [])
    with ir.InsertionPoint(loop3.body):
        iv = arith.index_cast(i32, loop3.induction_variable)
        col_v0 = gcu.vector_step(iv, vec_i32)
        mask0 = arith.cmpi(arith.CmpIPredicate.slt, col_v0, bcast_n)
        safe_iv0 = arith.minsi(iv, n_cols_m_vec)
        abs_off0 = arith.addi(row_off_i32, safe_iv0)
        mem_idx0 = arith.index_cast(idx, abs_off0)
        v0 = gcu.maskedload(in_memref, mem_idx0, mask0, vec_neg_inf, vec_f32)
        r0 = arith.mulf(math_d.exp(arith.subf(v0, bcast_max)), bcast_inv)
        gcu.maskedstore(out_memref, mem_idx0, mask0, r0)
        iv1 = arith.addi(iv, c_vec_i32)
        col_v1 = gcu.vector_step(iv1, vec_i32)
        mask1 = arith.cmpi(arith.CmpIPredicate.slt, col_v1, bcast_n)
        safe_iv1 = arith.minsi(iv1, n_cols_m_vec)
        abs_off1 = arith.addi(row_off_i32, safe_iv1)
        mem_idx1 = arith.index_cast(idx, abs_off1)
        v1 = gcu.maskedload(in_memref, mem_idx1, mask1, vec_neg_inf, vec_f32)
        r1 = arith.mulf(math_d.exp(arith.subf(v1, bcast_max)), bcast_inv)
        gcu.maskedstore(out_memref, mem_idx1, mask1, r1)
        scf.yield_([])


def _build_3pass_body_1v(in_memref, out_memref, vec_f32, vec_i32, vec_neg_inf, vec_zero, row_off_i32, bcast_n,
                         n_cols_m_vec, upper_idx, i32, f32, idx, c0_idx, c_vec_idx):
    """Emit the 3-pass softmax logic with single-vector per iteration (STEP=128).
    Lower register pressure allows larger BLOCK_SIZE for static unrolling."""
    neg_inf_f32 = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
    one_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 1.0))

    # === Pass 1: find max ===
    loop1 = scf.ForOp(c0_idx, upper_idx, c_vec_idx, [vec_neg_inf])
    with ir.InsertionPoint(loop1.body):
        acc = loop1.inner_iter_args[0]
        iv = arith.index_cast(i32, loop1.induction_variable)
        col_v = gcu.vector_step(iv, vec_i32)
        mask = arith.cmpi(arith.CmpIPredicate.slt, col_v, bcast_n)
        safe_iv = arith.minsi(iv, n_cols_m_vec)
        abs_off = arith.addi(row_off_i32, safe_iv)
        mem_idx = arith.index_cast(idx, abs_off)
        v = gcu.maskedload(in_memref, mem_idx, mask, vec_neg_inf, vec_f32)
        new_acc = arith.maximumf(acc, v)
        scf.yield_([new_acc])
    max_vec = loop1.results[0]
    scalar_max = vector_d.ReductionOp(f32, vector_d.CombiningKind.MAXIMUMF, max_vec).result
    bcast_max = gcu.vector_broadcast(scalar_max, vec_f32)

    # === Pass 2: accumulate sum = Σ exp(x-max) (read-only, no output writes) ===
    loop2 = scf.ForOp(c0_idx, upper_idx, c_vec_idx, [vec_zero])
    with ir.InsertionPoint(loop2.body):
        acc = loop2.inner_iter_args[0]
        iv = arith.index_cast(i32, loop2.induction_variable)
        col_v = gcu.vector_step(iv, vec_i32)
        mask = arith.cmpi(arith.CmpIPredicate.slt, col_v, bcast_n)
        safe_iv = arith.minsi(iv, n_cols_m_vec)
        abs_off = arith.addi(row_off_i32, safe_iv)
        mem_idx = arith.index_cast(idx, abs_off)
        v = gcu.maskedload(in_memref, mem_idx, mask, vec_neg_inf, vec_f32)
        exp_v = math_d.exp(arith.subf(v, bcast_max))
        new_acc = arith.addf(acc, exp_v)
        scf.yield_([new_acc])
    sum_vec = loop2.results[0]
    scalar_sum = vector_d.ReductionOp(f32, vector_d.CombiningKind.ADD, sum_vec).result
    inv_sum = arith.divf(one_f32, scalar_sum)
    bcast_inv = gcu.vector_broadcast(inv_sum, vec_f32)

    # === Pass 3: recompute exp(x-max)/sum and store to output ===
    loop3 = scf.ForOp(c0_idx, upper_idx, c_vec_idx, [])
    with ir.InsertionPoint(loop3.body):
        iv = arith.index_cast(i32, loop3.induction_variable)
        col_v = gcu.vector_step(iv, vec_i32)
        mask = arith.cmpi(arith.CmpIPredicate.slt, col_v, bcast_n)
        safe_iv = arith.minsi(iv, n_cols_m_vec)
        abs_off = arith.addi(row_off_i32, safe_iv)
        mem_idx = arith.index_cast(idx, abs_off)
        v = gcu.maskedload(in_memref, mem_idx, mask, vec_neg_inf, vec_f32)
        r = arith.mulf(math_d.exp(arith.subf(v, bcast_max)), bcast_inv)
        gcu.maskedstore(out_memref, mem_idx, mask, r)
        scf.yield_([])


def _make_edsl_static(BLOCK_SIZE: int):
    """Factory: EDSL softmax with compile-time constant loop bound (2-vector ILP)."""

    @dialect(name="tops_mlir", use_gcu_opt=True)
    def _edsl_softmax(
        output: Input[L["!gcu.ptr<f32>"]],
        input_ptr: Input[L["!gcu.ptr<f32>"]],
        n_cols: Input[L["i32"]],
        stride_elems: Input[L["i32"]],
    ):
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        f32 = ir.F32Type.get()
        idx = ir.IndexType.get()
        vec_f32 = ir.VectorType.get([VEC], f32)
        vec_i32 = ir.VectorType.get([VEC], i32)
        dyn = ir.ShapedType.get_dynamic_size()
        memref_dyn_f32 = ir.MemRefType.get([dyn], f32)

        c0_i32 = _ci32(i32, 0)
        c_vec_i32 = _ci32(i32, VEC)
        c0_idx = _cidx(idx, 0)
        c_step_idx = _cidx(idx, STEP)
        upper_idx = _cidx(idx, BLOCK_SIZE)

        neg_inf_f32 = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
        zero_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))

        in_memref = gcu.ptr2memref(input_ptr, memref_dyn_f32)
        out_memref = gcu.ptr2memref(output, memref_dyn_f32)
        vec_neg_inf = gcu.vector_broadcast(neg_inf_f32, vec_f32)
        vec_zero = gcu.vector_broadcast(zero_f32, vec_f32)

        pid = gpu.block_id(gpu.Dimension.x)
        pid_i32 = arith.index_cast(i32, pid)
        stride_i64 = arith.extsi(i64, stride_elems)
        pid_i64 = arith.extsi(i64, pid_i32)
        row_off_i32 = arith.trunci(i32, arith.muli(pid_i64, stride_i64))

        bcast_n = gcu.vector_broadcast(n_cols, vec_i32)
        n_cols_m_vec = arith.maxsi(arith.subi(n_cols, c_vec_i32), c0_i32)

        _build_3pass_body(in_memref, out_memref, vec_f32, vec_i32, vec_neg_inf, vec_zero, row_off_i32, bcast_n,
                          n_cols_m_vec, upper_idx, i32, f32, idx, c0_idx, c_step_idx)

    return _edsl_softmax


def _make_edsl_static_1v(BLOCK_SIZE: int):
    """Factory: single-vector per iteration (lower register pressure for larger sizes)."""

    @dialect(name="tops_mlir", use_gcu_opt=True)
    def _edsl_softmax_1v(
        output: Input[L["!gcu.ptr<f32>"]],
        input_ptr: Input[L["!gcu.ptr<f32>"]],
        n_cols: Input[L["i32"]],
        stride_elems: Input[L["i32"]],
    ):
        i32 = ir.IntegerType.get_signless(32)
        i64 = ir.IntegerType.get_signless(64)
        f32 = ir.F32Type.get()
        idx = ir.IndexType.get()
        vec_f32 = ir.VectorType.get([VEC], f32)
        vec_i32 = ir.VectorType.get([VEC], i32)
        dyn = ir.ShapedType.get_dynamic_size()
        memref_dyn_f32 = ir.MemRefType.get([dyn], f32)

        c0_i32 = _ci32(i32, 0)
        c_vec_i32 = _ci32(i32, VEC)
        c0_idx = _cidx(idx, 0)
        c_vec_idx = _cidx(idx, VEC)
        upper_idx = _cidx(idx, BLOCK_SIZE)

        neg_inf_f32 = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
        zero_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))

        in_memref = gcu.ptr2memref(input_ptr, memref_dyn_f32)
        out_memref = gcu.ptr2memref(output, memref_dyn_f32)
        vec_neg_inf = gcu.vector_broadcast(neg_inf_f32, vec_f32)
        vec_zero = gcu.vector_broadcast(zero_f32, vec_f32)

        pid = gpu.block_id(gpu.Dimension.x)
        pid_i32 = arith.index_cast(i32, pid)
        stride_i64 = arith.extsi(i64, stride_elems)
        pid_i64 = arith.extsi(i64, pid_i32)
        row_off_i32 = arith.trunci(i32, arith.muli(pid_i64, stride_i64))

        bcast_n = gcu.vector_broadcast(n_cols, vec_i32)
        n_cols_m_vec = arith.maxsi(arith.subi(n_cols, c_vec_i32), c0_i32)

        _build_3pass_body_1v(in_memref, out_memref, vec_f32, vec_i32, vec_neg_inf, vec_zero, row_off_i32, bcast_n,
                             n_cols_m_vec, upper_idx, i32, f32, idx, c0_idx, c_vec_idx)

    return _edsl_softmax_1v


# Dynamic kernel: 2-pass ONLINE softmax with 3-tier loop structure:
#   Pass 1 (Online): simultaneously compute max and sum in a single read pass.
#     Uses correction: d = d * exp(m_old - m_new) + exp(v - m_new)
#     Eliminates one full pass over data (3R+1W → 2R+1W = 25% less memory traffic).
#   Pass 2 (Normalize): exp(x - max) / sum → write output.
#   Each pass uses 3-tier loop:
#     Tier 1: 4-vec main loop (unmasked, 4*VEC=512 elems/iter, high ILP)
#     Tier 2: 1-vec middle loop (unmasked, VEC=128 elems/iter)
#     Tier 3: single masked tail (at most 1 maskedload/maskedstore per pass)
VEC_x2 = 2 * VEC  # 256
VEC_x3 = 3 * VEC  # 384
DYN4 = 4 * VEC  # 512


@dialect(name="tops_mlir", use_gcu_opt=True)
def _edsl_softmax_dyn(
    output: Input[L["!gcu.ptr<f32>"]],
    input_ptr: Input[L["!gcu.ptr<f32>"]],
    n_cols: Input[L["i32"]],
    stride_elems: Input[L["i32"]],
):
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    f32 = ir.F32Type.get()
    idx = ir.IndexType.get()
    vec_f32 = ir.VectorType.get([VEC], f32)
    vec_i32 = ir.VectorType.get([VEC], i32)
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_f32 = ir.MemRefType.get([dyn], f32)

    c0_i32 = _ci32(i32, 0)
    c_vec_i32 = _ci32(i32, VEC)
    c_4vec_i32 = _ci32(i32, DYN4)
    c0_idx = _cidx(idx, 0)
    c_vec_idx = _cidx(idx, VEC)
    c_4vec_idx = _cidx(idx, DYN4)
    c1v = _cidx(idx, VEC)
    c2v = _cidx(idx, VEC_x2)
    c3v = _cidx(idx, VEC_x3)

    neg_inf_f32 = arith.constant(f32, ir.FloatAttr.get(f32, NEG_INF))
    zero_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 0.0))
    one_f32 = arith.constant(f32, ir.FloatAttr.get(f32, 1.0))

    in_memref = gcu.ptr2memref(input_ptr, memref_dyn_f32)
    out_memref = gcu.ptr2memref(output, memref_dyn_f32)
    vec_neg_inf = gcu.vector_broadcast(neg_inf_f32, vec_f32)
    vec_zero = gcu.vector_broadcast(zero_f32, vec_f32)

    pid = gpu.block_id(gpu.Dimension.x)
    pid_i32 = arith.index_cast(i32, pid)
    stride_i64 = arith.extsi(i64, stride_elems)
    pid_i64 = arith.extsi(i64, pid_i32)
    row_off_i32 = arith.trunci(i32, arith.muli(pid_i64, stride_i64))
    row_off_idx = arith.index_cast(idx, row_off_i32)

    bcast_n = gcu.vector_broadcast(n_cols, vec_i32)

    n4_floor_i32 = arith.muli(arith.divsi(n_cols, c_4vec_i32), c_4vec_i32)
    n4_floor_idx = arith.index_cast(idx, n4_floor_i32)

    n1_floor_i32 = arith.muli(arith.divsi(n_cols, c_vec_i32), c_vec_i32)
    n1_floor_idx = arith.index_cast(idx, n1_floor_i32)

    remainder_i32 = arith.subi(n_cols, n1_floor_i32)
    has_tail = arith.cmpi(arith.CmpIPredicate.sgt, remainder_i32, c0_i32)

    # === Pass 1: Online max + sum (single read pass) ===
    # Carry: [d_vec (vector accumulator), m (scalar running max)]
    # Correction: when new_max > old_max, scale d by exp(old - new).
    lp1 = scf.ForOp(c0_idx, n4_floor_idx, c_4vec_idx, [vec_zero, neg_inf_f32])
    with ir.InsertionPoint(lp1.body):
        d_acc = lp1.inner_iter_args[0]
        m_acc = lp1.inner_iter_args[1]
        off = arith.addi(row_off_idx, lp1.induction_variable)
        v0 = vector_d.LoadOp(vec_f32, in_memref, [off]).result
        v1 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c1v)]).result
        v2 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c2v)]).result
        v3 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c3v)]).result
        m0123 = arith.maximumf(arith.maximumf(v0, v1), arith.maximumf(v2, v3))
        chunk_max = vector_d.ReductionOp(f32, vector_d.CombiningKind.MAXIMUMF, m0123).result
        new_m = arith.maximumf(m_acc, chunk_max)
        corr_vec = math_d.exp(gcu.vector_broadcast(arith.subf(m_acc, new_m), vec_f32))
        new_m_bcast = gcu.vector_broadcast(new_m, vec_f32)
        e0 = math_d.exp(arith.subf(v0, new_m_bcast))
        e1 = math_d.exp(arith.subf(v1, new_m_bcast))
        e2 = math_d.exp(arith.subf(v2, new_m_bcast))
        e3 = math_d.exp(arith.subf(v3, new_m_bcast))
        d_new = arith.addf(arith.mulf(d_acc, corr_vec), arith.addf(arith.addf(e0, e1), arith.addf(e2, e3)))
        scf.yield_([d_new, new_m])

    lp1m = scf.ForOp(n4_floor_idx, n1_floor_idx, c_vec_idx, [lp1.results[0], lp1.results[1]])
    with ir.InsertionPoint(lp1m.body):
        d_acc = lp1m.inner_iter_args[0]
        m_acc = lp1m.inner_iter_args[1]
        off = arith.addi(row_off_idx, lp1m.induction_variable)
        v = vector_d.LoadOp(vec_f32, in_memref, [off]).result
        chunk_max = vector_d.ReductionOp(f32, vector_d.CombiningKind.MAXIMUMF, v).result
        new_m = arith.maximumf(m_acc, chunk_max)
        corr_vec = math_d.exp(gcu.vector_broadcast(arith.subf(m_acc, new_m), vec_f32))
        new_m_bcast = gcu.vector_broadcast(new_m, vec_f32)
        e = math_d.exp(arith.subf(v, new_m_bcast))
        d_new = arith.addf(arith.mulf(d_acc, corr_vec), e)
        scf.yield_([d_new, new_m])

    p1_d = lp1m.results[0]
    p1_m = lp1m.results[1]
    if_tail1 = scf.IfOp(has_tail, [vec_f32, f32], hasElse=True)
    with ir.InsertionPoint(if_tail1.then_block):
        tail_off_idx = arith.index_cast(idx, arith.addi(row_off_i32, n1_floor_i32))
        col_v = gcu.vector_step(n1_floor_i32, vec_i32)
        mask = arith.cmpi(arith.CmpIPredicate.slt, col_v, bcast_n)
        tv = gcu.maskedload(in_memref, tail_off_idx, mask, vec_neg_inf, vec_f32)
        chunk_max = vector_d.ReductionOp(f32, vector_d.CombiningKind.MAXIMUMF, tv).result
        new_m = arith.maximumf(p1_m, chunk_max)
        corr_vec = math_d.exp(gcu.vector_broadcast(arith.subf(p1_m, new_m), vec_f32))
        new_m_bcast = gcu.vector_broadcast(new_m, vec_f32)
        e = math_d.exp(arith.subf(tv, new_m_bcast))
        masked_e = arith.select(mask, e, vec_zero)
        d_new = arith.addf(arith.mulf(p1_d, corr_vec), masked_e)
        scf.yield_([d_new, new_m])
    with ir.InsertionPoint(if_tail1.else_block):
        scf.yield_([p1_d, p1_m])

    scalar_sum = vector_d.ReductionOp(f32, vector_d.CombiningKind.ADD, if_tail1.results[0]).result
    final_m = if_tail1.results[1]
    inv_sum = arith.divf(one_f32, scalar_sum)
    bcast_inv = gcu.vector_broadcast(inv_sum, vec_f32)
    bcast_max = gcu.vector_broadcast(final_m, vec_f32)

    # === Pass 2: normalize + write ===
    lp2 = scf.ForOp(c0_idx, n4_floor_idx, c_4vec_idx, [])
    with ir.InsertionPoint(lp2.body):
        off = arith.addi(row_off_idx, lp2.induction_variable)
        v0 = vector_d.LoadOp(vec_f32, in_memref, [off]).result
        v1 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c1v)]).result
        v2 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c2v)]).result
        v3 = vector_d.LoadOp(vec_f32, in_memref, [arith.addi(off, c3v)]).result
        r0 = arith.mulf(math_d.exp(arith.subf(v0, bcast_max)), bcast_inv)
        r1 = arith.mulf(math_d.exp(arith.subf(v1, bcast_max)), bcast_inv)
        r2 = arith.mulf(math_d.exp(arith.subf(v2, bcast_max)), bcast_inv)
        r3 = arith.mulf(math_d.exp(arith.subf(v3, bcast_max)), bcast_inv)
        vector_d.StoreOp(r0, out_memref, [off])
        vector_d.StoreOp(r1, out_memref, [arith.addi(off, c1v)])
        vector_d.StoreOp(r2, out_memref, [arith.addi(off, c2v)])
        vector_d.StoreOp(r3, out_memref, [arith.addi(off, c3v)])
        scf.yield_([])

    lp2m = scf.ForOp(n4_floor_idx, n1_floor_idx, c_vec_idx, [])
    with ir.InsertionPoint(lp2m.body):
        off = arith.addi(row_off_idx, lp2m.induction_variable)
        v = vector_d.LoadOp(vec_f32, in_memref, [off]).result
        r = arith.mulf(math_d.exp(arith.subf(v, bcast_max)), bcast_inv)
        vector_d.StoreOp(r, out_memref, [off])
        scf.yield_([])

    if_tail2 = scf.IfOp(has_tail, [], hasElse=False)
    with ir.InsertionPoint(if_tail2.then_block):
        tail_off_idx = arith.index_cast(idx, arith.addi(row_off_i32, n1_floor_i32))
        col_v = gcu.vector_step(n1_floor_i32, vec_i32)
        mask = arith.cmpi(arith.CmpIPredicate.slt, col_v, bcast_n)
        tv = gcu.maskedload(in_memref, tail_off_idx, mask, vec_neg_inf, vec_f32)
        te = math_d.exp(arith.subf(tv, bcast_max))
        gcu.maskedstore(out_memref, tail_off_idx, mask, arith.mulf(te, bcast_inv))
        scf.yield_([])


# ---------------------------------------------------------------------------
# Pre-build static EDSL functions and Triton JIT wrappers
# ---------------------------------------------------------------------------
# <=1024: 2-vector ILP (STEP=256) with scf.for — proven stable.
# 1025..1280: single-vector (STEP=128) with scf.for — proven stable.
# >1280: L1-buffered dynamic kernel (memref.alloca) — 2N global, N exp.
# Static kernels for BLOCK_SIZE >= 1408 trigger SIP assert due to GCU
# compiler over-unrolling constant-bound loops.
MAX_STATIC = 1280

edsl_256 = _make_edsl_static(256)
edsl_512 = _make_edsl_static(512)
edsl_768 = _make_edsl_static(768)
edsl_1024 = _make_edsl_static(1024)
edsl_1152 = _make_edsl_static_1v(1152)
edsl_1280 = _make_edsl_static_1v(1280)


@triton.jit
def _k_s256(out, inp, n_cols, stride):
    tle_raw.call(edsl_256, [out, inp, n_cols, stride])


@triton.jit
def _k_s512(out, inp, n_cols, stride):
    tle_raw.call(edsl_512, [out, inp, n_cols, stride])


@triton.jit
def _k_s768(out, inp, n_cols, stride):
    tle_raw.call(edsl_768, [out, inp, n_cols, stride])


@triton.jit
def _k_s1024(out, inp, n_cols, stride):
    tle_raw.call(edsl_1024, [out, inp, n_cols, stride])


@triton.jit
def _k_s1152(out, inp, n_cols, stride):
    tle_raw.call(edsl_1152, [out, inp, n_cols, stride])


@triton.jit
def _k_s1280(out, inp, n_cols, stride):
    tle_raw.call(edsl_1280, [out, inp, n_cols, stride])


@triton.jit(do_not_specialize=[2, 3])
def _k_softmax_dyn(out, inp, n_cols, stride):
    tle_raw.call(_edsl_softmax_dyn, [out, inp, n_cols, stride])


_STATIC_MAP = {
    256: _k_s256,
    512: _k_s512,
    768: _k_s768,
    1024: _k_s1024,
    1152: _k_s1152,
    1280: _k_s1280,
}


def softmax_edsl(x):
    """EDSL softmax with hybrid dispatch:
    - N <= 1280: static kernels (compile-time loop bounds, max performance)
    - N > 1280: dynamic kernel (4-vec ILP main + masked tail, no loop overhead)
    """
    n_rows, n_cols = x.shape
    if n_cols <= 1024:
        block_size = ((n_cols + STEP - 1) // STEP) * STEP
        block_size = max(block_size, STEP)
    elif n_cols <= MAX_STATIC:
        block_size = ((n_cols + VEC - 1) // VEC) * VEC
    else:
        block_size = None
    y = torch.empty_like(x)
    stride = x.stride(0)
    if block_size is not None and block_size in _STATIC_MAP:
        _STATIC_MAP[block_size][(n_rows, )](y, x, n_cols, stride, num_warps=1)
    else:
        _k_softmax_dyn[(n_rows, )](y, x, n_cols, stride, num_warps=1)
    return y


# ---------------------------------------------------------------------------
# Triton Native Softmax (from openai-02-softmax.py golden)
# ---------------------------------------------------------------------------
@triton.jit
def _native_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax_native(x):
    """Triton native softmax (same as golden)."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    _native_softmax_kernel[(n_rows, )](y, x, x.stride(0), y.stride(0), n_cols, num_warps=4, BLOCK_SIZE=BLOCK_SIZE)
    return y


# ---------------------------------------------------------------------------
# Main: Correctness + Performance Benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    print("=" * 70)
    print("  Fused Softmax: Hybrid EDSL vs Triton Native vs Torch")
    print("=" * 70, flush=True)

    try:
        DEVICE = triton.runtime.driver.active.get_active_torch_device()
    except Exception:
        DEVICE = None

    if DEVICE is None:
        print("[No GCU device] Skipping runtime tests.")
        os._exit(0)

    print(f"  Device: {DEVICE}\n", flush=True)
    torch.manual_seed(0)

    # --- Correctness ---
    print("[Correctness Check]", flush=True)
    all_pass = True
    shapes = [(4, 256), (4, 384), (4, 512), (16, 1024), (4, 1280), (4, 2048), (32, 4096), (4, 8192), (4, 12672),
              (4096, 256), (4096, 1024), (4096, 4096)]

    for n_rows, n_cols in shapes:
        x = torch.randn(n_rows, n_cols, device=DEVICE, dtype=torch.float32)
        y_edsl = softmax_edsl(x)
        y_ref = torch.softmax(x, dim=1)
        ok = torch.allclose(y_edsl, y_ref, atol=1e-3, rtol=1e-3)
        if ok:
            print(f"  ({n_rows:>4}, {n_cols:>5}): PASS")
        else:
            diff = (y_edsl - y_ref).abs().max().item()
            print(f"  ({n_rows:>4}, {n_cols:>5}): FAIL  max_diff={diff:.6g}")
            all_pass = False

    if all_pass:
        print("\n  All correctness checks passed!")
    else:
        print("\n  *** SOME TESTS FAILED ***")

    # --- Benchmark ---
    M = 4096
    x_vals = [128 * i for i in range(2, 100)]

    print("\n[Benchmark] M={}, N={}..{}  (GB/s)\n".format(M, x_vals[0], x_vals[-1]), flush=True)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=x_vals,
            line_arg='provider',
            line_vals=['edsl', 'triton', 'torch'],
            line_names=['EDSL (GB/s)', 'Triton (GB/s)', 'Torch (GB/s)'],
            ylabel='GB/s',
            plot_name='gcu400-edsl-softmax-performance',
            args={'M': M},
        ))
    def benchmark(M, N, provider):
        x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
        if provider == 'torch':
            ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
        elif provider == 'triton':
            ms = triton.testing.do_bench(lambda: softmax_native(x))
        else:
            ms = triton.testing.do_bench(lambda: softmax_edsl(x))
        gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-6 / ms
        return gbps(ms)

    benchmark.run(print_data=True)
    print("\nDone.")
