# SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
# SPDX-License-Identifier: MIT
"""smt_rvisa built-in functions.

These are Python-side marker objects. When called inside a @smt_rvisa function
body, they get translated to MLIR ops by SmtMLIRCodeGenerator.

At Python runtime (outside codegen), they raise NotImplementedError so
accidental direct calls are caught early.
"""
from __future__ import annotations


class _SmtRvisaBuiltin:

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(f"smt_rvisa.{self._name}() must only be called inside a "
                                  f"@smt_rvisa function body (used by SmtMLIRCodeGenerator)")

    def __repr__(self):
        return f"smt_rvisa.{self._name}"


class _SmtRvisaRange:
    """Marker for smt_rvisa.range(...) — translated to scf.for bounds.

    Accepts range(stop) or range(start, stop, step) like the Python builtin;
    only ever evaluated by SmtMLIRCodeGenerator (raises if called directly).
    """

    def __call__(self, *args):
        raise NotImplementedError("smt_rvisa.range() must only be used in a @smt_rvisa function body")

    def __repr__(self):
        return "smt_rvisa.range"


# Public built-in objects
proton_mark = _SmtRvisaBuiltin("proton_mark")  # proton_mark(name, is_start) → rdtime + func.call @proton_record
range = _SmtRvisaRange()  # range(n) / range(start, stop, step) → scf.for bounds

# ---------------------------------------------------------------------------
# svector-level markers (feishu 3.3 mv example). Fixed-VL eDSL that maps
# document names to already-verified vector/arith/memref primitives.
#   vconfig  : vconfig(avl, lmul) → VLMAX const (SEW from dtype); avl deferred
#   vzero    : vector.broadcast 0.0 -> vector<VL x dtype>
#   vload    : transfer_read a VL-length vector (1D idx, or 2D idx + row stride)
#   vmacc    : widening multiply-accumulate  acc += extf(x) * extf(y)
#   vreduce_sum : vector.reduction <add> -> scalar
#   vstore   : store a scalar to memref[idx]
#   alloc    : memref.alloc N-D scratch (style3 packed_B)
#   pack     : pack a B row-block into the packed_B scratch layout (style3)
#   vpack    : vpack(v, group_len) → vector_ext.group_interleave → many smt.vpack.vv (cube interleave)
#   vmadot   : vmadot(acc, x, y) → vector_ext.cross_batch_matmul → many smt.vfwmadot (batched cube matmul)
#   vshape   : vshape(v, shape) → vector.shape_cast (reshape)
#   vbroadcast: vbroadcast(v, n) → vector.broadcast (broadcast dim)
# ---------------------------------------------------------------------------
vconfig = _SmtRvisaBuiltin(
    "vconfig")  # vconfig(avl, lmul) → VL = min(avl, VLMAX), VLMAX = lmul × VLEN / SEW (SPEC §6.1)
vzero = _SmtRvisaBuiltin("vzero")  # vzero(dtype) → vector<VL x dtype> zeros
vload = _SmtRvisaBuiltin("vload")  # vload(ptr, idx_tuple[, stride]) → vector<VL x dtype>
vmacc = _SmtRvisaBuiltin("vmacc")  # vmacc(acc, x, y) → widening fma accumulate
vreduce_sum = _SmtRvisaBuiltin("vreduce_sum")  # vreduce_sum(vec) → scalar
vreduce_max = _SmtRvisaBuiltin("vreduce_max")  # vreduce_max(vec) → scalar  (vector.reduction<maxf/maxsi>)
vreduce_min = _SmtRvisaBuiltin("vreduce_min")  # vreduce_min(vec) → scalar  (vector.reduction<minf/minsi>)
vreduce_mul = _SmtRvisaBuiltin("vreduce_mul")  # vreduce_mul(vec) → scalar  (vector.reduction<mul>)
vstore = _SmtRvisaBuiltin(
    "vstore")  # vstore(ptr, idx, vec) → transfer_write (vector only, width = VL; use sstore for scalars)
alloc = _SmtRvisaBuiltin("alloc")  # alloc(shape_tuple, dtype) → memref.alloc
pack = _SmtRvisaBuiltin("pack")  # pack(src, src_idx, dst, dst_shape) → pack rows (style3)
vpack = _SmtRvisaBuiltin(
    "vpack"
)  # vpack(v, group_len) → vector_ext.group_interleave → many smt.vpack.vv (cube interleave, vector<b×N>→<(b/2)×2N>)
vmadot = _SmtRvisaBuiltin(
    "vmadot")  # vmadot(acc, x, y) → vector_ext.cross_batch_matmul → many smt.vfwmadot (batched cube matmul)
vshape = _SmtRvisaBuiltin("vshape")  # vshape(v, shape) → vector.shape_cast (same-numel reshape)
vbroadcast = _SmtRvisaBuiltin(
    "vbroadcast")  # vbroadcast(v, n) → vector.broadcast: vector<64> → vector<n×64> (broadcast dim)
spread = _SmtRvisaBuiltin(
    "spread"
)  # spread(src, cube_shape=(kc,n,k)) → scf.for scalar-broadcast pack → memref<kc×(n*k)> (broadcasts A's n dim, bypassing vscale)
imin = _SmtRvisaBuiltin("imin")  # imin(a, b) → arith.minsi on index(valid_rows = imin(MB, M-row_base))
# §6.4 element-wise named functions (arithmetic operators use Python operators directly, no marker needed)
vmin = _SmtRvisaBuiltin("vmin")  # vmin(a, b) → element-wise min → arith.minimumf / minsi
vmax = _SmtRvisaBuiltin("vmax")  # vmax(a, b) → element-wise max → arith.maximumf / maxsi
sqrt = _SmtRvisaBuiltin("sqrt")  # sqrt(a) → √a → math.sqrt
rsqrt = _SmtRvisaBuiltin("rsqrt")  # rsqrt(a) → 1/√a → math.rsqrt
vexp = _SmtRvisaBuiltin("vexp")  # vexp(a)  → eˣ  → math.exp  (vector or scalar)
vlog = _SmtRvisaBuiltin("vlog")  # vlog(a)  → ln(a) → math.log (vector or scalar)
sload = _SmtRvisaBuiltin("sload")  # sload(ptr, idx, dtype=f32) → scalar load from ptr[idx] (s-prefix = scalar, no VL)
sstore = _SmtRvisaBuiltin("sstore")  # sstore(ptr, idx, scalar) → scalar store to ptr[idx] (s-prefix = scalar, no VL)
viota = _SmtRvisaBuiltin("viota")  # viota() → vector<VLxindex> [0,1,..,VL-1] (vector.step), for argmax index tracking
abs = _SmtRvisaBuiltin("abs")  # abs(a) → |a| → math.absf / absi  # noqa: A001 (shadows builtin intentionally)
cast = _SmtRvisaBuiltin("cast")  # cast(a, dtype) → type conversion → arith.extf/truncf/sitofp/fptosi/extsi/trunci
select = _SmtRvisaBuiltin("select")  # select(m, a, b) → a if m else b → arith.select (§6.8 fallback)

# ── LLVM-dialect llvm-direct primitives (full call_intrinsic kernel, all scalable) ──
call_intrinsic = _SmtRvisaBuiltin(
    "call_intrinsic")  # call_intrinsic(name, [ops], result_type=T) → llvm.call_intrinsic / dotted llvm op
llvm_poison = _SmtRvisaBuiltin("llvm_poison")  # llvm_poison(T) → llvm.mlir.poison : T (vle passthru)
llvm_const = _SmtRvisaBuiltin("llvm_const")  # llvm_const(v, T) → llvm.mlir.constant (scalar or dense splat)
llvm_base_ptr = _SmtRvisaBuiltin("llvm_base_ptr")  # llvm_base_ptr(mem) → llvm.extractvalue desc[1] → !llvm.ptr
llvm_gep = _SmtRvisaBuiltin("llvm_gep")  # llvm_gep(base, off, elem=) → llvm.getelementptr
llvm_size = _SmtRvisaBuiltin("llvm_size")  # llvm_size(mem, dim=) → llvm.extractvalue desc[3,dim] → i64

# ---------------------------------------------------------------------------
# Document-facing sugar: dtype names and the `mem` / `index` / `raw_kernel`
# helpers so a kernel can be written close to the feishu 3.3 surface syntax.
# dtype constants are plain MLIR element-type strings.
# ---------------------------------------------------------------------------
f16 = "f16"
f32 = "f32"
bf16 = "bf16"

# ---------------------------------------------------------------------------
# MMA cube size per dtype (K3, arch 0xA064). Mirrors spine-mlir's
# TargetDescriptionAnalysis::getMMACubicSize (MMACubicSize{m, n, k}):
#   f16/bf16 -> {8, 8, 8},  i8 -> {8, 16, 8},  i4 -> {8, 32, 8}
# so kernels derive spread/vmadot cube dims from the dtype instead of a
# hardcoded 8. Keyed by the eDSL element-type string.
# ---------------------------------------------------------------------------
_MMA_CUBE = {
    "f16": (8, 8, 8),
    "bf16": (8, 8, 8),
    "i8": (8, 16, 8),
    "i4": (8, 32, 8),
}


def mma_cube(dtype: str) -> tuple:
    """Return the (m, n, k) MMA cube size for `dtype` on K3.

    Mirrors TargetDescriptionAnalysis::getMMACubicSize so raw kernels can size
    spread/vmadot cubes from the target's MMA shape rather than a literal 8.
    """
    if dtype not in _MMA_CUBE:
        raise KeyError(f"no MMA cube size for dtype {dtype!r}; known: {sorted(_MMA_CUBE)}")
    return _MMA_CUBE[dtype]
