"""TLE (Triton Language Extensions) implementation of mhc_pre_clamp_sinkhorn.

1:1 port of the AscendC operator at
``/data/liuhy/ops-transformer/mhc/mhc_pre_clamp_sinkhorn`` (arch35 kernels in
``op_kernel/arch35/mhc_pre_clamp_sinkhorn_base_arch35.h``) written in Triton
using the TLE DSA surface documented under
``/data/liuhy/flir/flagtree/documents/tle``.

Semantics (both AscendC and this file)::

    x'      = x * inv_rms,   inv_rms = rsqrt(mean(x^2) + norm_eps)   # RMSNorm over hcMult*D
    mixes   = x' @ phi^T                                            # (T, hcMix)
    H^pre   = sigmoid(mixes[:, :N]   * alpha[0] + base[:N])   + hc_eps
    H^post  = 2 * sigmoid(mixes[:, N:2N] * alpha[1] + base[N:2N])
    logits  = mixes[:, 2N:] * alpha[2] + base[2N:]   -> (T, N, N)
    [clamp] logits -> clamp(logits, clamp_min, clamp_max)
    M       = softmax(logits, dim=-1) + hc_eps
    M      /= (M.sum(dim=-2, keepdim) + hc_eps)                    # col-norm (iter 0)
    for i in 1..iter_times-1:                                       # Sinkhorn
        M /= (M.sum(dim=-1, keepdim) + hc_eps)                     # row-norm
        M /= (M.sum(dim=-2, keepdim) + hc_eps)                     # col-norm
    h_in    = sum_n(x[t,n,:] * H^pre[t,n])                          # (T, D)

AscendC layout -> TLE mapping
-----------------------------
    ``CopyIn`` (DataCopyPad GM->UB, 1D / 2D with srcStride)
        ``tle.dsa.alloc([..], dtype, UB)`` + ``tle.dsa.copy(src_ptr, ub, [tail])``.
    UB->UB vector ops (``Cast``/``Muls``/``Add``/``Sigmoid``/``Adds``/``Exp``/
    ``ReduceMax``/``ReduceSum``/``Div``/``Sub``)
        ``tle.dsa.to_tensor(ub)`` -> ``tl`` expression -> ``tle.dsa.to_buffer(.., UB)``.
    Cube-side ``mm1_`` (x_scaled @ phi^T)
        ``torch.mm`` on device (the cube core path; not reimplemented in TLE).
    ``CopyOut`` (DataCopyPad UB->GM)
        ``tle.dsa.to_buffer(tensor, UB)`` + ``tle.dsa.copy(ub, dst_ptr, [tail])``.

Stage 1 (RMSNorm, ``VFProcessCastAndInvRmsPart1``)
    Two-pass fp32: pass 1 accumulates sum(x*x) and writes the cast x to UB;
    pass 2 multiplies by inv_rms. We mirror with a single TLE kernel that
    loops ``hcMult*D`` in ``BLOCK_H`` chunks, accumulating in an fp32 scalar,
    then writes x_scaled back in a second pass.

Stage 2 (heads + Sinkhorn, ``VFProcessPre``/``VFProcessPost``/
``VFProcessCombFragRLessVLUseFourUnfold``)
    One program per token. The 24 ``mixes`` and 24 ``base`` are staged in UB
    via a single 2D DSA copy, then turned into tensors (registers). The 16
    comb-logits stay in registers; clamp, row-softmax and ``iter_times`` rounds
    of Sinkhorn all run in registers -- the AscendC ``FourUnfold`` path keeps
    the 4 columns of the 4x4 matrix in 4 register tensors for the whole loop,
    which is exactly what the scalar unroll below expresses. ``iter_times`` is
    expanded with ``tl.static_range`` to match the AscendC
    ``for (int64_t iter = 1; iter < iterTimes; iter++)``.

Stage 3 (ProcessY, ``VFProcessY``)
    2D grid ``(T, ceil(D / BLOCK_D))``: each program reads ``pre[t, :4]`` once,
    then streams ``x[t, n, :]`` d-chunks from GM through UB into fp32 registers
    and accumulates ``hin = sum_n(pre[n] * x[n])`` before casting back to x's
    dtype and writing to GM.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle
    _HAS_DSA = hasattr(tle, "dsa") and hasattr(tle.dsa, "alloc")
except (ImportError, AttributeError):
    tle = None
    _HAS_DSA = False


# ===========================================================================
# Stage 1: RMSNorm + scale  (maps VFProcessCastAndInvRmsPart1)
#
# AscendC: per token, loop hcMult*D in VL_FP32(=64) chunks; pass 1 casts
# x->fp32, stores the cast to xCastLocal, and reduces sum(x*x) into a scalar;
# then inv_rms = rsqrt(sum * (1/d) + eps). We keep the same two-pass shape but
# use a single DSA-staged UB tile per chunk.
# ===========================================================================
@triton.jit
def _rms_scale_kernel_tle(
    x_ptr,  # (T, hcMult*D) input dtype
    x_scaled_ptr,  # (T, hcMult*D) fp32 output
    inv_rms_ptr,  # (T,) fp32
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,  # 1/HC_D
    NORM_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HC_D

    x_dt = x_ptr.dtype.element_ty

    # ---- pass 1: sum(x*x) in fp32, stage each chunk through UB ----
    # NOTE: a plain Python float literal (0.0) is used for the scalar fp32
    # accumulator instead of ``tl.zeros([], dtype=tl.float32)``; the latter
    # triggers a triton-ascend compiler segfault once 3+ scalar-tensor
    # accumulators are live in the same program (see mhc_post_backward.py).
    sq = 0.0
    x_ub = tle.dsa.alloc([BLOCK_H], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        tail = tl.minimum(HC_D - h_start, BLOCK_H)
        with tle.dsa.hint(inter_no_alias=True):
            tle.dsa.copy(x_ptr + base + offs, x_ub, [tail])
        v = tle.dsa.to_tensor(x_ub).to(tl.float32)
        # mask-out the tail in fp32 so the padded UB lanes do not pollute sq
        v = tl.where(mask, v, 0.0)
        sq += tl.sum(v * v, axis=0)

    inv = tl.math.rsqrt(sq * D_INV + NORM_EPS)
    tl.store(inv_rms_ptr + pid, inv)

    # ---- pass 2: x_scaled = x * inv_rms ----
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        tail = tl.minimum(HC_D - h_start, BLOCK_H)
        with tle.dsa.hint(inter_no_alias=True):
            tle.dsa.copy(x_ptr + base + offs, x_ub, [tail])
        v = tle.dsa.to_tensor(x_ub).to(tl.float32)
        v = tl.where(mask, v, 0.0)
        out = v * inv
        out_buf = tle.dsa.to_buffer(out, tle.dsa.ascend.UB)
        with tle.dsa.hint(inter_no_alias=True):
            tle.dsa.copy(out_buf, x_scaled_ptr + base + offs, [tail])


# ===========================================================================
# Stage 2: pre / post / combLogits + clamp + softmax + Sinkhorn
#   (maps VFProcessPre / VFProcessPost / VFProcessCombFragRLessVLUseFourUnfold)
#
# One program per token. The 24 mixes + 24 base are staged in UB then moved to
# registers (tl.tensor scalars). The 16 comb-logits live in registers for the
# whole clamp -> row-softmax -> (iter_times-1) x (row-norm, col-norm) loop,
# exactly like the AscendC FourUnfold path that keeps mix1..mix4 in registers
# across the iteration loop.
# ===========================================================================
@triton.jit
def _heads_sinkhorn_kernel_tle(mixes_ptr,  # (T, 24) fp32
                               alpha_ptr,  # (3,)   fp32
                               base_ptr,  # (24,)  fp32
                               pre_ptr,  # (T, 4)        fp32  OUT
                               post_ptr,  # (T, 4)        fp32  OUT
                               comb_ptr,  # (T, 4, 4)     fp32  OUT
                               logits_ptr,  # (T, 4, 4)     fp32  OUT (pre-clamp logits, for bwd)
                               HC_EPS: tl.constexpr, CLAMP_MIN: tl.constexpr, CLAMP_MAX: tl.constexpr,
                               APPLY_CLAMP: tl.constexpr, ITERS: tl.constexpr, SAVE_INTERMEDIATES: tl.constexpr,
                               NUM_TOKENS: tl.constexpr,  # tokens per program (pipeline depth)
                               ):
    """Pipeline version: each program processes NUM_TOKENS tokens sequentially.

    Uses tle.dsa.pipeline(num_stages=2) to overlap MTE2 DMA (loading next
    token's mixes from GM->UB) with Vector compute (sigmoid + Sinkhorn on
    current token). Mirrors the mhc_post pipeline pattern.

    Grid: (cdiv(T, NUM_TOKENS),)
    """
    pid = tl.program_id(0)
    token_start = pid * NUM_TOKENS

    # ---- Load constants: alpha (3 scalars), base (24 values) ----
    # base is shared across all tokens; load once via DSA bulk copy into UB.
    base_ub = tle.dsa.alloc([32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(base_ptr + tl.arange(0, 32), base_ub, [24])
    base_vec = tle.dsa.to_tensor(base_ub)  # (32,) fp32, only [0:24] valid

    # Extract base sub-vectors via extract_slice (contiguous UB slices)
    base_pre = tl.reshape(tle.dsa.extract_slice(base_vec, (0, ), (4, ), (1, )), [4])
    base_po = tl.reshape(tle.dsa.extract_slice(base_vec, (4, ), (4, ), (1, )), [4])
    base_l0 = tl.reshape(tle.dsa.extract_slice(base_vec, (8, ), (4, ), (1, )), [4])
    base_l1 = tl.reshape(tle.dsa.extract_slice(base_vec, (12, ), (4, ), (1, )), [4])
    base_l2 = tl.reshape(tle.dsa.extract_slice(base_vec, (16, ), (4, ), (1, )), [4])
    base_l3 = tl.reshape(tle.dsa.extract_slice(base_vec, (20, ), (4, ), (1, )), [4])

    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)

    # ---- Allocate double-buffered UB for mixes[24] per token ----
    # tle.dsa.pipeline with num_stages=2: while computing token N, DMA loads token N+1.
    mix_ub = tle.dsa.alloc([32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined token loop ----
    for t_local in tle.dsa.pipeline(0, NUM_TOKENS, 1, num_stages=2):
        t_idx = token_start + t_local
        mb = t_idx * 24

        # -- DMA stage: copy mixes[t_idx, :24] from GM to UB --
        tle.dsa.copy(mixes_ptr + mb + tl.arange(0, 32), mix_ub, [24])

        # -- Compute stage: extract mixes sub-vectors from UB --
        mix_vec = tle.dsa.to_tensor(mix_ub)  # (32,) fp32
        mix_pre = tl.reshape(tle.dsa.extract_slice(mix_vec, (0, ), (4, ), (1, )), [4])
        mix_po = tl.reshape(tle.dsa.extract_slice(mix_vec, (4, ), (4, ), (1, )), [4])
        mix_l0 = tl.reshape(tle.dsa.extract_slice(mix_vec, (8, ), (4, ), (1, )), [4])
        mix_l1 = tl.reshape(tle.dsa.extract_slice(mix_vec, (12, ), (4, ), (1, )), [4])
        mix_l2 = tl.reshape(tle.dsa.extract_slice(mix_vec, (16, ), (4, ), (1, )), [4])
        mix_l3 = tl.reshape(tle.dsa.extract_slice(mix_vec, (20, ), (4, ), (1, )), [4])

        # ---- pre head: sigmoid(mix*a0 + base) + hc_eps ---- vectorized (4,)
        pre_vec = tl.sigmoid(mix_pre * a0 + base_pre) + HC_EPS

        # ---- post head: 2 * sigmoid(mix*a1 + base) ---- vectorized (4,)
        post_vec = 2.0 * tl.sigmoid(mix_po * a1 + base_po)

        # ---- combLogits: mix*a2 + base ---- vectorized per row (4,)
        r0 = mix_l0 * a2 + base_l0
        r1 = mix_l1 * a2 + base_l1
        r2 = mix_l2 * a2 + base_l2
        r3 = mix_l3 * a2 + base_l3

        # ---- save pre-clamp logits (vectorized store) ----
        if SAVE_INTERMEDIATES:
            lb = t_idx * 16
            offs4 = tl.arange(0, 4)
            tl.store(logits_ptr + lb + offs4, r0)
            tl.store(logits_ptr + lb + 4 + offs4, r1)
            tl.store(logits_ptr + lb + 8 + offs4, r2)
            tl.store(logits_ptr + lb + 12 + offs4, r3)

        # ---- optional clamp ---- vectorized per row (4,)
        if APPLY_CLAMP:
            r0 = tl.minimum(tl.maximum(r0, CLAMP_MIN), CLAMP_MAX)
            r1 = tl.minimum(tl.maximum(r1, CLAMP_MIN), CLAMP_MAX)
            r2 = tl.minimum(tl.maximum(r2, CLAMP_MIN), CLAMP_MAX)
            r3 = tl.minimum(tl.maximum(r3, CLAMP_MIN), CLAMP_MAX)

        # ---- row-softmax: max -> sub -> exp -> sum -> div ---- vectorized (4,)
        m0 = tl.max(r0, axis=0)
        m1 = tl.max(r1, axis=0)
        m2 = tl.max(r2, axis=0)
        m3 = tl.max(r3, axis=0)
        e0 = tl.exp(r0 - m0)
        e1 = tl.exp(r1 - m1)
        e2 = tl.exp(r2 - m2)
        e3 = tl.exp(r3 - m3)
        s0 = tl.sum(e0, axis=0)
        s1 = tl.sum(e1, axis=0)
        s2 = tl.sum(e2, axis=0)
        s3 = tl.sum(e3, axis=0)
        r0 = e0 / s0
        r1 = e1 / s1
        r2 = e2 / s2
        r3 = e3 / s3

        # ---- iter 0 col-norm: M = softmax + hc_eps; M /= (colsum + hc_eps) ----
        r0 = r0 + HC_EPS
        r1 = r1 + HC_EPS
        r2 = r2 + HC_EPS
        r3 = r3 + HC_EPS
        col_sum = r0 + r1 + r2 + r3 + HC_EPS
        r0 = r0 / col_sum
        r1 = r1 / col_sum
        r2 = r2 / col_sum
        r3 = r3 / col_sum

        # ---- remaining (ITERS-1) Sinkhorn iterations ----
        for _ in tl.static_range(ITERS - 1):
            rs0 = tl.sum(r0, axis=0) + HC_EPS
            rs1 = tl.sum(r1, axis=0) + HC_EPS
            rs2 = tl.sum(r2, axis=0) + HC_EPS
            rs3 = tl.sum(r3, axis=0) + HC_EPS
            r0 = r0 / rs0
            r1 = r1 / rs1
            r2 = r2 / rs2
            r3 = r3 / rs3
            cs = r0 + r1 + r2 + r3 + HC_EPS
            r0 = r0 / cs
            r1 = r1 / cs
            r2 = r2 / cs
            r3 = r3 / cs

        # ---- store results via vectorized store ----
        offs4 = tl.arange(0, 4)
        pb = t_idx * 4
        tl.store(pre_ptr + pb + offs4, pre_vec)
        tl.store(post_ptr + pb + offs4, post_vec)
        cb = t_idx * 16
        tl.store(comb_ptr + cb + offs4, r0)
        tl.store(comb_ptr + cb + 4 + offs4, r1)
        tl.store(comb_ptr + cb + 8 + offs4, r2)
        tl.store(comb_ptr + cb + 12 + offs4, r3)


# ===========================================================================
# Stage 3: y = sum_n(x[n] * pre[n])  (maps VFProcessY)
#
# 2D grid (T, cdiv(D, BLOCK_D)). Each program loads pre[t,:4] once (scalars),
# then streams the 4 x[D]-rows for this token through UB in BLOCK_D chunks,
# casts to fp32, accumulates hin = sum_n(pre[n]*x[n]) and writes back in x's
# dtype. Mirrors VFProcessY's per-token, per-d-chunk LoadInputDataWithBrc +
# Mul + Add + StoreOutputData loop.
# ===========================================================================
@triton.jit
def _y_scale_kernel_tle(
    x_ptr,  # (T, 4, D) input dtype
    pre_ptr,  # (T, 4)    fp32
    y_ptr,  # (T, D)    input dtype
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    tail_d = tl.minimum(D - pid_d * BLOCK_D, BLOCK_D)

    # pre[t, :4] scalars (AscendC loads these once per token via brc)
    p0 = tl.load(pre_ptr + pid_t * 4 + 0)
    p1 = tl.load(pre_ptr + pid_t * 4 + 1)
    p2 = tl.load(pre_ptr + pid_t * 4 + 2)
    p3 = tl.load(pre_ptr + pid_t * 4 + 3)

    x_dt = x_ptr.dtype.element_ty
    xb = pid_t * 4 * D

    # stage the 4 rows of this d-chunk into one (4, BLOCK_D) UB tile, mirroring
    # the AscendC 2D DataCopyPad with blockCount=4, srcStride=(D-dNum)*sizeof(T)
    n_idx = tl.arange(0, 4)
    src_2d = x_ptr + xb + n_idx[:, None] * D + d_off[None, :]
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])

    x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)
    # extract the 4 rows; mask tail lanes to 0 so they don't affect the sum
    x0 = tl.reshape(tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    x1 = tl.reshape(tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    x2 = tl.reshape(tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    x3 = tl.reshape(tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    x0 = tl.where(d_mask, x0, 0.0)
    x1 = tl.where(d_mask, x1, 0.0)
    x2 = tl.where(d_mask, x2, 0.0)
    x3 = tl.where(d_mask, x3, 0.0)

    hin = x0 * p0 + x1 * p1 + x2 * p2 + x3 * p3
    out_buf = tle.dsa.to_buffer(hin.to(x_dt), tle.dsa.ascend.UB)
    y_off = pid_t * D + d_off
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(out_buf, y_ptr + y_off, [tail_d])


# ===========================================================================
# Row-store fallback variant of stage 3.
#
# The pure-DSA kernel above is the literal 1:1 mapping of VFProcessY, but the
# triton-ascend compiler on this branch mis-orders the (4, BLOCK_D) tile built
# from extract_slice chains (same class of issue documented in mhc_post.py).
# This variant keeps the identical scalar-register data flow (pre hoisted per
# token, fp32 accumulators, single fp32->T cast on store) but issues the four
# head loads/stores through the regular masked path. Numerics are identical;
# only the copy-out DMA differs.
# ===========================================================================
@triton.jit
def _y_scale_kernel_tle_rows(
    x_ptr,  # (T, 4, D) input dtype
    pre_ptr,  # (T, 4)    fp32
    y_ptr,  # (T, D)    input dtype
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    p0 = tl.load(pre_ptr + pid_t * 4 + 0)
    p1 = tl.load(pre_ptr + pid_t * 4 + 1)
    p2 = tl.load(pre_ptr + pid_t * 4 + 2)
    p3 = tl.load(pre_ptr + pid_t * 4 + 3)

    xb = pid_t * 4 * D
    x0 = tl.load(x_ptr + xb + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + xb + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + xb + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + xb + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    hin = x0 * p0 + x1 * p1 + x2 * p2 + x3 * p3
    dt = y_ptr.dtype.element_ty
    tl.store(y_ptr + pid_t * D + d_off, hin.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _flatten(x):
    """Return (xf_TND, y_out_shape) where y_out_shape is the shape of the
    reduced-N `hin` output (aclnn semantic): (B, S, D) for 4D input or
    (T, D) for 3D input."""
    if x.dim() == 4:
        B, S, N, D = x.shape
        return x.reshape(B * S, N, D).contiguous(), (B, S, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return x.contiguous(), (T, D)
    raise ValueError(f"unsupported x.dim()={x.dim()}")


def mhc_pre_clamp_sinkhorn(
    x: torch.Tensor,
    phi: torch.Tensor,
    alpha: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
    iter_times: int = 20,
    need_backward: bool = False,
):
    """Fused MHC pre + clamp + Sinkhorn forward (aclnn semantic).

    Returns a dict with:
        y (hin)     : (B, S, D) or (T, D)  same dtype as x
                      hin[t, d] = sum_n( x[t, n, d] * pre[t, n] )
        post_out    : (T, hcMult)   fp32   post_out = 2 * sigmoid(...)
        comb_frag   : (T, hcMult, hcMult) fp32
    If need_backward=True, also:
        inv_rms     : (T,)  fp32
        x_scaled    : (T, hcMult*D) fp32
        mixes       : (T, hcMix) fp32
        h_res_logits: (T, hcMult, hcMult) fp32  (pre-clamp logits)
        pre         : (T, hcMult) fp32
    """
    if not _HAS_DSA:
        raise RuntimeError("This mhc_pre_clamp_sinkhorn implementation requires the TLE DSA "
                           "surface (triton.experimental.tle.dsa.*).")

    xf, shape = _flatten(x)
    T, N, D = xf.shape
    assert N == 4, "hc_mult=4 fast path only (matches AscendC hcMult=4)."
    hc_mix = N * (N + 2)
    hc_d = N * D
    assert phi.shape == (hc_mix, hc_d), f"phi shape {phi.shape} != {(hc_mix, hc_d)}"
    assert alpha.numel() == 3
    assert base.numel() == hc_mix

    x_scaled = torch.empty(T, hc_d, dtype=torch.float32, device=xf.device)
    inv_rms = torch.empty(T, dtype=torch.float32, device=xf.device)

    # Stage 1: RMSNorm + scale. BLOCK_H picked so the UB tile stays small and
    # T * (hc_d / BLOCK_H) stays well under the 65535 coreDim cap.
    BLOCK_H = min(1024, triton.next_power_of_2(hc_d))
    while T * triton.cdiv(hc_d, BLOCK_H) > 65535 and BLOCK_H < hc_d:
        BLOCK_H *= 2
    BLOCK_H = min(BLOCK_H, triton.next_power_of_2(hc_d))
    _rms_scale_kernel_tle[(T, )](
        xf.view(T, hc_d),
        x_scaled,
        inv_rms,
        HC_D=hc_d,
        D_INV=1.0 / hc_d,
        NORM_EPS=norm_eps,
        BLOCK_H=BLOCK_H,
    )

    # Cube-side GEMM: mixes = x_scaled @ phi^T. AscendC runs this on the cube
    # core (mm1_.Init/Process); torch.mm dispatches to the same MMAD path.
    phi_f = phi.to(torch.float32)
    mixes = torch.mm(x_scaled, phi_f.t())  # (T, hcMix)

    pre = torch.empty(T, N, dtype=torch.float32, device=xf.device)
    post_out = torch.empty(T, N, dtype=torch.float32, device=xf.device)
    comb_frag = torch.empty(T, N, N, dtype=torch.float32, device=xf.device)
    h_res_logits = (torch.empty(T, N, N, dtype=torch.float32, device=xf.device) if need_backward else torch.empty(
        0, device=xf.device))

    apply_clamp = 1 if (clamp_min != 0.0 or clamp_max != 0.0) else 0
    # Pipeline kernel: each program processes NUM_TOKENS tokens with DMA/compute
    # overlap via tle.dsa.pipeline(num_stages=2). Pick NUM_TOKENS to balance
    # pipeline depth vs grid parallelism.
    NUM_TOKENS_PER_PROG = min(8, T)
    grid_heads = (triton.cdiv(T, NUM_TOKENS_PER_PROG), )
    _heads_sinkhorn_kernel_tle[grid_heads](
        mixes,
        alpha.to(torch.float32),
        base.to(torch.float32),
        pre,
        post_out,
        comb_frag,
        h_res_logits,
        HC_EPS=hc_eps,
        CLAMP_MIN=float(clamp_min),
        CLAMP_MAX=float(clamp_max),
        APPLY_CLAMP=apply_clamp,
        ITERS=int(iter_times),
        SAVE_INTERMEDIATES=1 if need_backward else 0,
        NUM_TOKENS=NUM_TOKENS_PER_PROG,
    )

    y = torch.empty(T, D, dtype=xf.dtype, device=xf.device)
    BLOCK_D = min(1024, triton.next_power_of_2(D))
    while T * triton.cdiv(D, BLOCK_D) > 65535 and BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_D = min(BLOCK_D, triton.next_power_of_2(D))
    grid_y = (T, triton.cdiv(D, BLOCK_D))
    # See NOTE ON KERNEL SELECTION in mhc_post.py: the pure-DSA kernel
    # (_y_scale_kernel_tle) is the literal 1:1 mapping of VFProcessY, but the
    # triton-ascend compiler mis-orders the (4, BLOCK_D) tile built from
    # extract_slice chains. We run the row-store variant, which keeps the same
    # scalar-register data flow but issues the four head loads/stores through
    # the regular masked path. Numerics identical; only the DMA differs.
    _y_scale_kernel_tle_rows[grid_y](xf, pre, y, D=D, BLOCK_D=BLOCK_D)

    result = {
        "y": y.reshape(shape),
        "post_out": post_out,
        "comb_frag": comb_frag,
    }
    if need_backward:
        result.update(
            inv_rms=inv_rms,
            x_scaled=x_scaled,
            mixes=mixes,
            h_res_logits=h_res_logits,
            pre=pre,
        )
    return result


def mhc_pre_clamp_sinkhorn_ref(
    x,
    phi,
    alpha,
    base,
    norm_eps=1e-6,
    hc_eps=1e-6,
    clamp_min=0.0,
    clamp_max=0.0,
    iter_times=20,
):
    """PyTorch reference implementation (aclnn semantic).

    hin[t, d] = sum_n( x[t, n, d] * pre[t, n] )   -> shape (B, S, D)
    post_out  = 2 * sigmoid(...)                  per aclnn spec.
    """
    orig_dtype = x.dtype
    xf, shape = _flatten(x)
    T, N, D = xf.shape
    x_flat = xf.reshape(T, N * D).float()

    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + norm_eps)
    x_scaled = x_flat * inv
    mixes = x_scaled @ phi.float().t()

    a = alpha.float()
    b = base.float()
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    post_out = 2.0 * torch.sigmoid(mixes[:, N:2 * N] * a[1] + b[N:2 * N])
    logits = (mixes[:, 2 * N:] * a[2] + b[2 * N:]).reshape(T, N, N)
    if clamp_min != 0.0 or clamp_max != 0.0:
        logits_c = torch.clamp(logits, clamp_min, clamp_max)
    else:
        logits_c = logits
    row_max = logits_c.max(dim=-1, keepdim=True).values
    M = (logits_c - row_max).exp()
    M = M / M.sum(dim=-1, keepdim=True) + hc_eps
    M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iter_times - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + hc_eps)
        M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)

    # hin = sum_n (x * pre) -> (T, D)
    y = (xf.float() * pre.unsqueeze(-1)).sum(dim=-2).to(orig_dtype)
    return {
        "y": y.reshape(shape),
        "post_out": post_out,
        "comb_frag": M,
        "inv_rms": inv.squeeze(-1),
        "mixes": mixes,
        "h_res_logits": logits,
        "pre": pre,
    }


__all__ = [
    "mhc_pre_clamp_sinkhorn",
    "mhc_pre_clamp_sinkhorn_ref",
    "_rms_scale_kernel_tle",
    "_heads_sinkhorn_kernel_tle",
    "_y_scale_kernel_tle",
    "_y_scale_kernel_tle_rows",
]
