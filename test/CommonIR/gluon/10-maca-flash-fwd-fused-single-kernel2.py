"""C500 FlashAttention forward with guarded D128 M64/M128 specialization.

The kernel implements dense and causal D128 attention for BHSD tensors.  The
causal FP16 specialization owns one M64 query tile and follows production
``flash_fwd_kernel_hdim128_blockM64.h``.  All other cases preserve the proven
M128 implementation.  The M64 physical shared arena is reused in three epochs::

    prologue:  sQ[64, 128] | unused[64, 128]
    main loop: sK[64, 128] | sV[64, 128]
    epilogue:  sO[64, 128] | unused[64, 128]

Q is loaded through registers, materialized in shared, and reloaded as the
persistent MMA A operand before sQ is overwritten by K.  For BM64, Q/O use the
C-aligned [M, 2, 64] D64-paged shared view; BM128 retains the generic D128
qkv layout.  K and V retain the generic register-prefetch/local-store flow.
Both MMA stages use [4, 1] warp ownership, so the two N32 QK accumulators feed
PV-A by trivial register retags without duplicate N-warp ownership.

The main physical difference from C is the V consumer.  C performs byte_perm
before shared and consumes an lds_Tuple=2 BSM fragment.  The explicit Gluon BSM
contract fixes a 32-register operand that currently forces [2, 2] ownership;
this kernel instead uses an ordinary Dot-B local load from two N32 shared
fragments.  A single N64 QK accumulator would match C more closely, but its two
N32 PV projections are not lowerable, so QK remains two N32 dots.
"""

import argparse
import math
import warnings

import torch
import triton

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward
except (ImportError, OSError) as error:
    _flash_attn_forward = None
    _FLASH_ATTN_IMPORT_ERROR = str(error)
else:
    _FLASH_ATTN_IMPORT_ERROR = None

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from maca_compose_shared_layout import (
    compose_maca_shared_layout_for_operand,
)


BM64 = 64
BM128 = 128
BN = 64
D = 128
NUM_WARPS = 4
HEADS = 32
LOG2E = 1.4426950408889634
TEST_SEQ_LENS = (128, 257)


@gluon.jit
def _flash_fwd_tile(
    n_block,
    k_register,
    acc_o,
    m_i,
    l_i,
    q_for_score,
    q_m,
    k_head,
    v_head,
    k_stride_n,
    k_stride_d,
    v_stride_n,
    v_stride_d,
    SQ,
    SK,
    scale_log2,
    first_d,
    k_tn_smem,
    v_fragments,
    tn_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    MASKED: gl.constexpr,
    CAUSAL: gl.constexpr,
):
    # Keep one K memdesc ABI across the tt.func boundary. Derive fragment
    # subviews in the callee so the whole-tile store is a visible covering
    # definition for both local loads.
    k_tn_fragments = k_tn_smem._reinterpret(
        k_head.dtype.element_ty,
        [BLOCK_N // 32, BLOCK_D, 32],
        tn_layout,
    )
    current_n = n_block * BLOCK_N + gl.arange(0, BLOCK_N)
    n0 = n_block * BLOCK_N + gl.arange(0, 32)
    n1 = n_block * BLOCK_N + 32 + gl.arange(0, 32)

    k0_smem = k_tn_fragments.index(0)
    k1_smem = k_tn_fragments.index(1)
    if BLOCK_M == 64:
        # M64 needs exact fragment writes because shared definite-init cannot
        # currently prove whole-reinterpret coverage of indexed child loads.
        k0_smem.store(
            gl.metax.slice(k_register, [32, BLOCK_D], [0, 0]).permute(1, 0)
        )
        k1_smem.store(
            gl.metax.slice(k_register, [32, BLOCK_D], [32, 0]).permute(1, 0)
        )
    else:
        # Preserve the established M128 whole-K path, including BF16 lowering.
        k_tn_smem.store(k_register.permute(1, 0))
    current_v_ptrs = (
        v_head
        + current_n[:, None] * v_stride_n
        + first_d[None, :] * v_stride_d
    )
    if MASKED:
        v_register = gl.load(
            current_v_ptrs,
            mask=current_n[:, None] < SK,
            other=0.0,
        )
    else:
        v_register = gl.load(current_v_ptrs)

    gl.metax.barrier_shared()
    kt0_for_score = k0_smem.load(intrinsic=True, is_constant_offs=True)
    score0 = gl.dot(
        q_for_score,
        kt0_for_score,
        gl.zeros((BLOCK_M, 32), gl.float32),
        input_precision="tf32",
    )
    kt1_for_score = k1_smem.load(intrinsic=True, is_constant_offs=True)
    score1 = gl.dot(
        q_for_score,
        kt1_for_score,
        gl.zeros((BLOCK_M, 32), gl.float32),
        input_precision="tf32",
    )

    if MASKED:
        point_mask0 = (q_m[:, None] < SQ) & (n0[None, :] < SK)
        point_mask1 = (q_m[:, None] < SQ) & (n1[None, :] < SK)
        if CAUSAL:
            causal_limit = q_m[:, None] + SK - SQ
            point_mask0 = point_mask0 & (n0[None, :] <= causal_limit)
            point_mask1 = point_mask1 & (n1[None, :] <= causal_limit)
        score0 = gl.where(point_mask0, score0 * scale_log2, -float("inf"))
        score1 = gl.where(point_mask1, score1 * scale_log2, -float("inf"))
    else:
        score0 = score0 * scale_log2
        score1 = score1 * scale_log2

    # Match copy_reg_to_share_V before the next-K prefetch.  Keep each N32
    # fragment in the ordinary Dot-B shared contract consumed below.
    v_fragments.index(0).store(
        gl.metax.slice(v_register, [32, BLOCK_D], [0, 0])
    )
    v_fragments.index(1).store(
        gl.metax.slice(v_register, [32, BLOCK_D], [32, 0])
    )

    has_next = n_block > 0
    next_n_block = max(n_block - 1, 0)
    next_n = next_n_block * BLOCK_N + gl.arange(0, BLOCK_N)
    next_valid = has_next & (next_n[:, None] < SK)
    next_k_ptrs = (
        k_head
        + next_n[:, None] * k_stride_n
        + first_d[None, :] * k_stride_d
    )
    next_k_register = gl.load(next_k_ptrs, mask=next_valid, other=0.0)

    tile_max = gl.maximum(gl.max(score0, axis=1), gl.max(score1, axis=1))
    gl.metax.barrier_shared()
    m_ij = gl.maximum(m_i, tile_max)
    if MASKED:
        row_has_key = (q_m < SQ) & (m_ij != -float("inf"))
        stable_m = gl.where(row_has_key, m_ij, 0.0)
        p0_qk = gl.where(
            point_mask0,
            gl.exp2(score0 - stable_m[:, None]),
            0.0,
        )
        p1_qk = gl.where(
            point_mask1,
            gl.exp2(score1 - stable_m[:, None]),
            0.0,
        )
        alpha = gl.where(row_has_key, gl.exp2(m_i - stable_m), 0.0)
    else:
        row_has_key = m_ij != -float("inf")
        stable_m = m_ij
        p0_qk = gl.exp2(score0 - stable_m[:, None])
        p1_qk = gl.exp2(score1 - stable_m[:, None])
        alpha = gl.exp2(m_i - stable_m)

    l_i = l_i * alpha + gl.sum(p0_qk, axis=1) + gl.sum(p1_qk, axis=1)
    acc_o = acc_o * alpha[:, None]
    m_i = gl.where(row_has_key, m_ij, m_i)

    # QK-C is a semantic producer of PV-A. The compiler derives both physical
    # dot contracts jointly and materializes only the required use-site
    # register transfer; source code does not prescribe an MMA fragment.
    p0 = p0_qk.to(v_head.dtype.element_ty)
    p1 = p1_qk.to(v_head.dtype.element_ty)
    v00 = v_fragments.index(0).load(intrinsic=True, is_constant_offs=True)
    acc_o = gl.dot(p0, v00, acc_o, input_precision="tf32")

    v10 = v_fragments.index(1).load(intrinsic=True, is_constant_offs=True)
    acc_o = gl.dot(p1, v10, acc_o, input_precision="tf32")
    return next_k_register, acc_o, m_i, l_i


@gluon.jit
def flash_fwd_hdim128_generic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SQ,
    SK,
    H: gl.constexpr,
    q_stride_b,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_n,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_n,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_m,
    o_stride_d,
    lse_stride_b,
    lse_stride_h,
    lse_stride_m,
    scale_log2,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    CAUSAL: gl.constexpr,
    EVEN_MN: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_bh = gl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    o_head = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h

    qk_elements_mnk: gl.constexpr = [1, 2, 8]
    pv_elements_mnk: gl.constexpr = [1, 2, 8]

    # Exactly 32 KiB for fp16/bf16. Q/O use buffer 0 in their epochs, while
    # K and V use disjoint buffers 0 and 1 in the main-loop epoch.
    # The M128 fallback keeps the established flat-D128 shared layout.
    qkv_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        qk_elements_mnk,
        0,
        [BLOCK_M, BLOCK_D],
        [1, 0],
        16,
    )
    # Production M64 Q/O uses Cute Swizzle<3,3,3> independently in each D64
    # page.  Reshaping [M,2,64] preserves
    #   page*4096 + m*64 + (((d64//8) ^ (m%8))*8 + d64%8).
    qo_d64_page_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[2, 0, 1]
    )
    v_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        pv_elements_mnk,
        1,
        [BLOCK_N // 2, BLOCK_D],
        [1, 0],
        16,
    )
    # QK-B is [K128, N32].  MACA's shared-operand composition for tk=8,
    # tn=2 and fp16 yields vec=8, perPhase=2, maxPhase=8 with K contiguous.
    tn_layout: gl.constexpr = compose_maca_shared_layout_for_operand(
        qk_elements_mnk,
        1,
        [BLOCK_D, BLOCK_N // 2],
        [0, 1],
        16,
    )
    shared_arena = gl.local_alloc(
        q_ptr.dtype.element_ty,
        [BLOCK_N, BLOCK_D],
        num_buffers=2,
    )
    if BLOCK_M == 64:
        qo_d64_pages = shared_arena.index(0)._reinterpret(
            q_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D // 64, 64],
            qo_d64_page_layout,
        )
        q_smem = qo_d64_pages.reshape([BLOCK_M, BLOCK_D])
        o_smem = q_smem
    else:
        q_smem = shared_arena._reinterpret(
            q_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D],
            qkv_layout,
        )
        o_smem = shared_arena._reinterpret(
            o_ptr.dtype.element_ty,
            [BLOCK_M, BLOCK_D],
            qkv_layout,
        )
    k_tn_smem = shared_arena.index(0)._reinterpret(
        k_ptr.dtype.element_ty,
        [BLOCK_D, BLOCK_N],
        tn_layout,
    )
    v_fragments = shared_arena.index(1)._reinterpret(
        v_ptr.dtype.element_ty,
        [BLOCK_N // 32, 32, BLOCK_D],
        v_layout,
    )
    q_load_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    q_load_d = gl.arange(0, BLOCK_D)
    q_ptrs = (
        q_head
        + q_load_m[:, None] * q_stride_m
        + q_load_d[None, :] * q_stride_d
    )
    if EVEN_MN:
        q_register = gl.load(q_ptrs)
    else:
        q_register = gl.load(q_ptrs, mask=q_load_m[:, None] < SQ, other=0.0)

    q_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)

    num_n_blocks = gl.cdiv(SK, BLOCK_N)
    if CAUSAL:
        causal_hi = (pid_m + 1) * BLOCK_M + SK - SQ
        active_n_blocks = min(num_n_blocks, max(0, gl.cdiv(causal_hi, BLOCK_N)))
    else:
        active_n_blocks = num_n_blocks
    first_n_block = active_n_blocks - 1

    first_n = first_n_block * BLOCK_N + gl.arange(0, BLOCK_N)
    first_d = gl.arange(0, BLOCK_D)
    first_k_ptrs = (
        k_head
        + first_n[:, None] * k_stride_n
        + first_d[None, :] * k_stride_d
    )
    if EVEN_MN and not CAUSAL:
        k_register = gl.load(first_k_ptrs)
    else:
        first_valid = (first_n_block >= 0) & (first_n[:, None] < SK)
        k_register = gl.load(first_k_ptrs, mask=first_valid, other=0.0)

    # C prologue: gQ -> rQ -> sQ, then publish and retain the MMA Q fragment.
    q_smem.store(q_register)
    gl.metax.barrier_shared()
    q_for_score = q_smem.load(intrinsic=True, is_constant_offs=True)

    # Every thread must finish reading sQ before the first K tile aliases it.
    gl.metax.barrier_shared()

    acc_o = gl.zeros((BLOCK_M, BLOCK_D), gl.float32)
    m_i = gl.full((BLOCK_M,), -float("inf"), gl.float32)
    l_i = gl.zeros((BLOCK_M,), gl.float32)

    if CAUSAL:
        if EVEN_MN:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N)
        else:
            masking_steps: gl.constexpr = gl.cdiv(BLOCK_M, BLOCK_N) + 1
    elif EVEN_MN:
        masking_steps: gl.constexpr = 0
    else:
        masking_steps: gl.constexpr = 1

    masked_blocks = min(active_n_blocks, masking_steps)
    # The even causal M64 specialization executes C's first masked tile as a
    # straight-line stage. Clamp the block only for active=0 CTAs; its causal
    # point mask is then all false. Uneven M64 retains a dynamic guard, while
    # M128 keeps its original range(0, masked_blocks) structure below.
    if BLOCK_M == 64 and CAUSAL and EVEN_MN:
        n_block = max(first_n_block, 0)
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block,
            k_register,
            acc_o,
            m_i,
            l_i,
            q_for_score,
            q_m,
            k_head,
            v_head,
            k_stride_n,
            k_stride_d,
            v_stride_n,
            v_stride_d,
            SQ,
            SK,
            scale_log2,
            first_d,
            k_tn_smem,
            v_fragments,
            tn_layout,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            True,
            CAUSAL,
        )
    elif BLOCK_M == 64 and masked_blocks > 0:
        n_block = first_n_block
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block,
            k_register,
            acc_o,
            m_i,
            l_i,
            q_for_score,
            q_m,
            k_head,
            v_head,
            k_stride_n,
            k_stride_d,
            v_stride_n,
            v_stride_d,
            SQ,
            SK,
            scale_log2,
            first_d,
            k_tn_smem,
            v_fragments,
            tn_layout,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            True,
            CAUSAL,
        )

    if BLOCK_M == 64:
        masked_loop_begin: gl.constexpr = 1
    else:
        masked_loop_begin: gl.constexpr = 0
    for n_iteration in range(masked_loop_begin, masked_blocks):
        n_block = first_n_block - n_iteration
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block,
            k_register,
            acc_o,
            m_i,
            l_i,
            q_for_score,
            q_m,
            k_head,
            v_head,
            k_stride_n,
            k_stride_d,
            v_stride_n,
            v_stride_d,
            SQ,
            SK,
            scale_log2,
            first_d,
            k_tn_smem,
            v_fragments,
            tn_layout,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            True,
            CAUSAL,
        )

    for n_iteration in range(masked_blocks, active_n_blocks):
        n_block = first_n_block - n_iteration
        k_register, acc_o, m_i, l_i = _flash_fwd_tile(
            n_block,
            k_register,
            acc_o,
            m_i,
            l_i,
            q_for_score,
            q_m,
            k_head,
            v_head,
            k_stride_n,
            k_stride_d,
            v_stride_n,
            v_stride_d,
            SQ,
            SK,
            scale_log2,
            first_d,
            k_tn_smem,
            v_fragments,
            tn_layout,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            False,
            CAUSAL,
        )

    row_has_key = (q_m < SQ) & (l_i > 0.0)
    safe_l = gl.where(row_has_key, l_i, 1.0)
    out = gl.where(
        row_has_key[:, None],
        acc_o / safe_l[:, None],
        0.0,
    )
    lse = gl.where(
        row_has_key,
        (m_i + gl.log2(safe_l)) * 0.6931471805599453,
        float("inf"),
    )

    # C epilogue: the complete arena changes ownership from K|V to O.  Stage O
    # through shared so the final global writer has the coalesced Q/O layout.
    gl.metax.barrier_shared()
    o_smem.store(out.to(o_ptr.dtype.element_ty))
    gl.metax.barrier_shared()
    # Leave the register owner open.  Candidate construction must satisfy both
    # the fixed shared-memory reader map and the D-contiguous global store; this
    # is the complete physical boundary that the user previously had to encode
    # as a hand-written BlockedLayout.
    out_coalesced = o_smem.load(intrinsic=True, is_constant_offs=True)

    # The shared round-trip changes ownership from the PV accumulator to a
    # D-contiguous global writer.  Rebuild offsets and mask in that ownership
    # domain instead of propagating the QK row layout into the final store.
    o_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
    o_d = gl.arange(0, BLOCK_D)
    o_ptrs = (
        o_head
        + o_m[:, None] * o_stride_m
        + o_d[None, :] * o_stride_d
    )
    gl.store(o_ptrs, out_coalesced, mask=o_m[:, None] < SQ)
    gl.store(
        lse_head + q_m * lse_stride_m,
        lse,
        mask=q_m < SQ,
    )


def _validate_inputs(q, k, v):
    if q.ndim != 4:
        raise ValueError("Q must be a BHSD tensor")
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim != D:
        raise ValueError(f"head dimension must be {D}, got {head_dim}")
    if k.ndim != 4 or k.shape[:2] != (batch, heads) or k.shape[3] != D:
        raise ValueError(f"invalid K shape {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if seqlen_q <= 0 or k.shape[2] <= 0:
        raise ValueError("query and key sequence lengths must be positive")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, and V must have the same dtype")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Q, K, and V must be contiguous in the head dimension")
    if any(t.device != q.device for t in (k, v)):
        raise ValueError("Q, K, and V must be on the same device")
    return batch, heads, seqlen_q, k.shape[2]


def launch_flash_fwd(q, k, v, out=None, lse=None, scale=None, causal=False):
    batch, heads, seqlen_q, seqlen_k = _validate_inputs(q, k, v)
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(
            (batch, heads, seqlen_q),
            device=q.device,
            dtype=torch.float32,
        )
    if out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("output must match Q and be contiguous in head dimension")
    if lse.shape != (batch, heads, seqlen_q) or lse.dtype != torch.float32:
        raise ValueError("LSE must be fp32 [B,H,SQ]")
    if not lse.is_contiguous():
        raise ValueError("LSE must be contiguous")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    block_m = BM64 if causal and q.dtype == torch.float16 else BM128
    grid = (triton.cdiv(seqlen_q, block_m), batch * heads)
    flash_fwd_hdim128_generic_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        seqlen_q,
        seqlen_k,
        heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        float(scale * LOG2E),
        BLOCK_M=block_m,
        BLOCK_N=BN,
        BLOCK_D=D,
        CAUSAL=causal,
        # Mirrors C's Is_even_MN specialization.  Only proven full tiles can
        # select the dense noncausal load/score path without point masks.
        EVEN_MN=(seqlen_q % block_m == 0 and seqlen_k % BN == 0),
        num_warps=NUM_WARPS,
        num_stages=2,
        num_ctas=1,
        pipeline="cpasync",
    )
    return out, lse


def _prepare_flash_attn_forward(q, k, v, scale, causal):
    if _flash_attn_forward is None:
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        if causal:
            seq_q, seq_k = scores.shape[-2:]
            causal_mask = torch.ones(
                (seq_q, seq_k), device=scores.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)
        return out, lse, None

    # flash_attn consumes BSHD tensors. Keep layout conversion outside both
    # measured regions so the table compares device attention calls directly.
    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    def launch():
        return _flash_attn_forward(
            q_fa,
            k_fa,
            v_fa,
            0.0,
            scale,
            causal,
            (-1, -1),
        )

    (
        _,
        _,
        _,
        _,
        out_fa,
        lse_fa,
        _,
        _,
        _,
    ) = launch()
    return out_fa.transpose(1, 2), lse_fa, launch


def _run_case(seq_len, dtype, causal, warmup, rep):
    torch.manual_seed(0)
    shape = (1, HEADS, seq_len, D)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(D)
    out, lse = launch_flash_fwd(q, k, v, scale=scale, causal=causal)
    flash_out, flash_lse, flash_launch = _prepare_flash_attn_forward(
        q, k, v, scale, causal
    )
    result = {
        "allclose": torch.allclose(
            out, flash_out, atol=3e-2, rtol=3e-2
        )
        and torch.allclose(lse, flash_lse, atol=3e-2, rtol=3e-2),
    }
    result["gluon_ms"] = triton.testing.do_bench(
        lambda: launch_flash_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            scale=scale,
            causal=causal,
        ),
        warmup=warmup,
        rep=rep,
    )
    if flash_launch is None:
        result["flash_ms"] = None
        result["speedup_vs_flash"] = None
    else:
        result["flash_ms"] = triton.testing.do_bench(
            flash_launch,
            warmup=warmup,
            rep=rep,
        )
        result["speedup_vs_flash"] = result["flash_ms"] / result["gluon_ms"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", nargs="+", type=int, default=TEST_SEQ_LENS)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if _flash_attn_forward is None:
        warnings.warn(
            "flash_attn is unavailable; using a PyTorch FP32 correctness "
            f"reference and omitting flash_attn timing: {_FLASH_ATTN_IMPORT_ERROR}",
            RuntimeWarning,
        )

    rows = []
    results = []
    for seq_len in tuple(dict.fromkeys(args.seq_len)):
        result = _run_case(
            seq_len, dtype, args.causal, args.warmup, args.rep
        )
        results.append(result)
        row = [
            f"1x{HEADS}x{seq_len}x{D}",
            dtype,
            args.causal,
            f"{result['gluon_ms']:.6f}",
            (
                f"{result['flash_ms']:.6f}"
                if result["flash_ms"] is not None
                else "unavailable"
            ),
            (
                f"{result['speedup_vs_flash']:.3f}"
                if result["speedup_vs_flash"] is not None
                else "unavailable"
            ),
            result["allclose"],
        ]
        rows.append(row)

    headers = [
        "BxHxSxD",
        "dtype",
        "causal",
        "gluon_ms",
        "_flash_attn_forward_ms",
        "gluon_speedup_vs_flash",
        "accuracy_ok",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(value) for value in row) + " |")
    if not all(result["allclose"] for result in results):
        raise AssertionError(f"FlashAttention forward accuracy failure: {rows}")


if __name__ == "__main__":
    main()
