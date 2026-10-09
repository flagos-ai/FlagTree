"""TLE/CommonIR port of the basic MACA fused FlashAttention forward.

Matches gluon/10-maca-flash-fwd-fused-single-kernel.py's descending KV loop,
register K prefetch, two N32 score fragments, online softmax and output/LSE.
Register and shared-memory layouts are inferred; no source layout is fixed.
Unlike the Gluon arena, Q/K/V/O use separate buffers: physical memory reuse
is left to the compiler. Supports FP16/BF16, D64/D128 and bottom-right causal
masking, including unequal or non-tile-aligned query/key lengths.

    MACA_VISIBLE_DEVICES=6 python test/CommonIR/metax/10-maca-flash-fwd-fused-single-kernel.py
"""

import argparse
import math

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._common_ir import ENABLED

BM64 = 64
BM128 = 128
BN = 64
SUPPORTED_HEAD_DIMS = (64, 128)
NUM_WARPS = 4
LOG2E = 1.4426950408889634


@triton.jit
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
    k_fragments,
    v_fragments,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MASKED: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    current_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n0 = n_block * BLOCK_N + tl.arange(0, 32)
    n1 = n_block * BLOCK_N + 32 + tl.arange(0, 32)

    k0_smem = k_fragments.slot(0)
    k1_smem = k_fragments.slot(1)
    k0_smem.store(tle.extract_tile(k_register, (0, 0), [32, BLOCK_D]).permute(1, 0))
    k1_smem.store(tle.extract_tile(k_register, (1, 0), [32, BLOCK_D]).permute(1, 0))
    current_v_ptrs = (v_head + current_n[:, None] * v_stride_n + first_d[None, :] * v_stride_d)
    if MASKED:
        v_register = tl.load(
            current_v_ptrs,
            mask=current_n[:, None] < SK,
            other=0.0,
        )
    else:
        v_register = tl.load(current_v_ptrs)

    tle.gpu.metax.barrier_shared()
    kt0_for_score = tle.gpu.metax.local_load(k0_smem, intrinsic=True, is_constant_offs=True)
    score0 = tl.dot(
        q_for_score,
        kt0_for_score,
        tl.zeros((BLOCK_M, 32), tl.float32),
        input_precision="tf32",
    )
    kt1_for_score = tle.gpu.metax.local_load(k1_smem, intrinsic=True, is_constant_offs=True)
    score1 = tl.dot(
        q_for_score,
        kt1_for_score,
        tl.zeros((BLOCK_M, 32), tl.float32),
        input_precision="tf32",
    )

    if MASKED:
        point_mask0 = (q_m[:, None] < SQ) & (n0[None, :] < SK)
        point_mask1 = (q_m[:, None] < SQ) & (n1[None, :] < SK)
        if CAUSAL:
            causal_limit = q_m[:, None] + SK - SQ
            point_mask0 = point_mask0 & (n0[None, :] <= causal_limit)
            point_mask1 = point_mask1 & (n1[None, :] <= causal_limit)
        score0 = tl.where(point_mask0, score0 * scale_log2, -float("inf"))
        score1 = tl.where(point_mask1, score1 * scale_log2, -float("inf"))
    else:
        score0 = score0 * scale_log2
        score1 = score1 * scale_log2

    # Match copy_reg_to_share_V before the next-K prefetch.  Keep each N32
    # fragment in the ordinary Dot-B shared contract consumed below.
    v_fragments.slot(0).store(tle.extract_tile(v_register, (0, 0), [32, BLOCK_D]))
    v_fragments.slot(1).store(tle.extract_tile(v_register, (1, 0), [32, BLOCK_D]))

    has_next = n_block > 0
    next_n_block = max(n_block - 1, 0)
    next_n = next_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    next_valid = has_next & (next_n[:, None] < SK)
    next_k_ptrs = (k_head + next_n[:, None] * k_stride_n + first_d[None, :] * k_stride_d)
    next_k_register = tl.load(next_k_ptrs, mask=next_valid, other=0.0)

    tile_max = tl.max(tl.maximum(score0, score1), axis=1)
    tle.gpu.metax.barrier_shared()
    m_ij = tl.maximum(m_i, tile_max)
    if MASKED:
        row_has_key = (q_m < SQ) & (m_ij != -float("inf"))
        stable_m = tl.where(row_has_key, m_ij, 0.0)
        p0_qk = tl.where(
            point_mask0,
            tl.exp2(score0 - stable_m[:, None]),
            0.0,
        )
        p1_qk = tl.where(
            point_mask1,
            tl.exp2(score1 - stable_m[:, None]),
            0.0,
        )
        alpha = tl.where(row_has_key, tl.exp2(m_i - stable_m), 0.0)
    else:
        row_has_key = m_ij != -float("inf")
        stable_m = m_ij
        p0_qk = tl.exp2(score0 - stable_m[:, None])
        p1_qk = tl.exp2(score1 - stable_m[:, None])
        alpha = tl.exp2(m_i - stable_m)

    l_i = l_i * alpha + tl.sum(p0_qk + p1_qk, axis=1)
    acc_o = acc_o * alpha[:, None]
    m_i = tl.where(row_has_key, m_ij, m_i)

    # QK-C is a semantic producer of PV-A. The compiler derives both physical
    # dot contracts jointly and materializes only the required use-site
    # register transfer; source code does not prescribe an MMA fragment.
    p0 = p0_qk.to(v_head.dtype.element_ty)
    p1 = p1_qk.to(v_head.dtype.element_ty)
    v00 = tle.gpu.metax.local_load(v_fragments.slot(0), intrinsic=True, is_constant_offs=True)
    acc_o = tl.dot(p0, v00, acc_o, input_precision="tf32")

    v10 = tle.gpu.metax.local_load(v_fragments.slot(1), intrinsic=True, is_constant_offs=True)
    acc_o = tl.dot(p1, v10, acc_o, input_precision="tf32")
    return next_k_register, acc_o, m_i, l_i


@triton.jit(autolayout=True)
def flash_fwd_hdim128_generic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    SQ,
    SK,
    H: tl.constexpr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    EVEN_MN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    q_head = q_ptr + pid_b * q_stride_b + pid_h * q_stride_h
    k_head = k_ptr + pid_b * k_stride_b + pid_h * k_stride_h
    v_head = v_ptr + pid_b * v_stride_b + pid_h * v_stride_h
    o_head = o_ptr + pid_b * o_stride_b + pid_h * o_stride_h
    lse_head = lse_ptr + pid_b * lse_stride_b + pid_h * lse_stride_h

    # CommonIR selects every shared layout from its consumers. Separate
    # buffers replace Gluon's explicit lifetime-disjoint arena aliases.
    q_smem = tle.gpu.alloc((BLOCK_M, BLOCK_D), q_ptr.dtype.element_ty)

    q_load_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_load_d = tl.arange(0, BLOCK_D)
    q_ptrs = (q_head + q_load_m[:, None] * q_stride_m + q_load_d[None, :] * q_stride_d)
    if EVEN_MN:
        q_register = tl.load(q_ptrs)
    else:
        q_register = tl.load(q_ptrs, mask=q_load_m[:, None] < SQ, other=0.0)

    q_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    num_n_blocks = tl.cdiv(SK, BLOCK_N)
    if CAUSAL:
        causal_hi = (pid_m + 1) * BLOCK_M + SK - SQ
        active_n_blocks = min(num_n_blocks, max(0, tl.cdiv(causal_hi, BLOCK_N)))
    else:
        active_n_blocks = num_n_blocks
    first_n_block = active_n_blocks - 1

    first_n = first_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    first_d = tl.arange(0, BLOCK_D)
    first_k_ptrs = (k_head + first_n[:, None] * k_stride_n + first_d[None, :] * k_stride_d)
    if EVEN_MN and not CAUSAL:
        k_register = tl.load(first_k_ptrs)
    else:
        first_valid = (first_n_block >= 0) & (first_n[:, None] < SK)
        k_register = tl.load(first_k_ptrs, mask=first_valid, other=0.0)

    # C prologue: gQ -> rQ -> sQ, then publish and retain the MMA Q fragment.
    q_smem.store(q_register)
    tle.gpu.metax.barrier_shared()
    q_for_score = tle.gpu.metax.local_load(q_smem, intrinsic=True, is_constant_offs=True)

    # Finish the Q shared-memory reads before the main loop.
    tle.gpu.metax.barrier_shared()

    # Start K/V buffer lifetimes after Q shared-memory reads have finished.
    k_fragments = tle.gpu.alloc((2, BLOCK_D, 32), k_ptr.dtype.element_ty)
    v_fragments = tle.gpu.alloc((2, 32, BLOCK_D), v_ptr.dtype.element_ty)

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M, ), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M, ), tl.float32)

    if CAUSAL:
        if EVEN_MN:
            masking_steps: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N)
        else:
            masking_steps: tl.constexpr = tl.cdiv(BLOCK_M, BLOCK_N) + 1
    elif EVEN_MN:
        masking_steps: tl.constexpr = 0
    else:
        masking_steps: tl.constexpr = 1

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
            k_fragments,
            v_fragments,
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
            k_fragments,
            v_fragments,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            True,
            CAUSAL,
        )

    if BLOCK_M == 64:
        masked_loop_begin: tl.constexpr = 1
    else:
        masked_loop_begin: tl.constexpr = 0
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
            k_fragments,
            v_fragments,
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
            k_fragments,
            v_fragments,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            False,
            CAUSAL,
        )

    row_has_key = (q_m < SQ) & (l_i > 0.0)
    safe_l = tl.where(row_has_key, l_i, 1.0)
    out = tl.where(
        row_has_key[:, None],
        acc_o / safe_l[:, None],
        0.0,
    )
    lse = tl.where(
        row_has_key,
        (m_i + tl.log2(safe_l)) * 0.6931471805599453,
        float("inf"),
    )

    # Retain the source's shared-memory output staging; infer its layout.
    o_smem = tle.gpu.alloc((BLOCK_M, BLOCK_D), o_ptr.dtype.element_ty)
    tle.gpu.metax.barrier_shared()
    o_smem.store(out.to(o_ptr.dtype.element_ty))
    tle.gpu.metax.barrier_shared()
    # Leave the register owner open.  Candidate construction must satisfy both
    # the shared-memory reader map and the D-contiguous global store; this
    # is the complete physical boundary that the user previously had to encode
    # as a hand-written BlockedLayout.
    out_coalesced = tle.gpu.metax.local_load(o_smem, intrinsic=True, is_constant_offs=True)

    # The shared round-trip changes ownership from the PV accumulator to a
    # D-contiguous global writer.  Rebuild offsets and mask in that ownership
    # domain instead of propagating the QK row layout into the final store.
    o_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    o_d = tl.arange(0, BLOCK_D)
    o_ptrs = (o_head + o_m[:, None] * o_stride_m + o_d[None, :] * o_stride_d)
    tl.store(o_ptrs, out_coalesced, mask=o_m[:, None] < SQ)
    tl.store(
        lse_head + q_m * lse_stride_m,
        lse,
        mask=q_m < SQ,
    )


def _validate_inputs(q, k, v):
    if not ENABLED:
        raise RuntimeError("Requires a MetaX FLAGTREE_COMMON_IR build")
    if not q.is_cuda:
        raise ValueError("Q, K, and V must be GPU tensors")
    if q.ndim != 4:
        raise ValueError("Q must be a BHSD tensor")
    batch, heads, seqlen_q, head_dim = q.shape
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head dimension must be in {SUPPORTED_HEAD_DIMS}, got {head_dim}")
    if k.ndim != 4 or k.shape[:2] != (batch, heads) or k.shape[3] != head_dim:
        raise ValueError(f"invalid K shape {tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError("V must have the same shape as K")
    if batch <= 0 or heads <= 0 or seqlen_q <= 0 or k.shape[2] <= 0:
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
    head_dim = q.shape[-1]
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
    if out.device != q.device or lse.device != q.device:
        raise ValueError("output and LSE must be on the input device")
    if not lse.is_contiguous():
        raise ValueError("LSE must be contiguous")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

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
        BLOCK_D=head_dim,
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


def reference_attention(q, k, v, scale=None, causal=False):
    """FP32 reference with bounded score storage and bottom-right causal masking."""
    scale = q.shape[-1]**-0.5 if scale is None else scale
    seq_q, seq_k = q.shape[-2], k.shape[-2]
    outputs, log_sums = [], []
    k_float = k.float().transpose(-1, -2)
    v_float = v.float()
    columns = torch.arange(seq_k, device=q.device)
    for start in range(0, seq_q, 128):
        stop = min(start + 128, seq_q)
        scores = torch.matmul(q[:, :, start:stop].float(), k_float) * scale
        if causal:
            rows = torch.arange(start, stop, device=q.device)
            mask = columns[None, :] <= rows[:, None] + seq_k - seq_q
            scores.masked_fill_(~mask, -float("inf"))
        lse = torch.logsumexp(scores, dim=-1)
        valid = torch.isfinite(lse)
        probabilities = torch.exp(scores - torch.where(valid, lse, 0.0)[..., None])
        outputs.append(torch.matmul(probabilities, v_float).to(q.dtype))
        log_sums.append(torch.where(valid, lse, float("inf")))
    return torch.cat(outputs, dim=-2), torch.cat(log_sums, dim=-1)


def _make_inputs(batch, heads, seq_q, seq_k, dimension, dtype):
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn((batch, heads, seq_q, dimension), device="cuda", dtype=dtype, generator=generator)
    k, v = [
        torch.randn((batch, heads, seq_k, dimension), device="cuda", dtype=dtype, generator=generator) for _ in range(2)
    ]
    return q, k, v


def _check(q, k, v, causal, scale=None):
    out, lse = launch_flash_fwd(q, k, v, causal=causal, scale=scale)
    expected, expected_lse = reference_attention(q, k, v, causal=causal, scale=scale)
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-2, rtol=3e-2)
    finite = torch.isfinite(expected_lse)
    errors = (float((out - expected).abs().max()), float((lse[finite] - expected_lse[finite]).abs().max()))
    return out, lse, errors


def _is_maca():
    try:
        return ENABLED and triton.runtime.driver.active.get_current_target().backend == "maca"
    except RuntimeError:
        return False


@pytest.mark.skipif(not _is_maca(), reason="Requires MetaX/MACA with CommonIR")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dimension", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq_q,seq_k", [(128, 128), (257, 257), (129, 65), (65, 129)])
def test_flash_fwd(dtype, dimension, causal, seq_q, seq_k):
    # Includes tail tiles, unequal lengths, and completely masked causal rows.
    _check(*_make_inputs(1, 2, seq_q, seq_k, dimension, dtype), causal)


@pytest.mark.skipif(not _is_maca(), reason="Requires MetaX/MACA with CommonIR")
def test_flash_fwd_strided_batch_and_outputs():
    q, k, v = [x[:, ::2, ::2, :] for x in _make_inputs(2, 4, 130, 258, 64, torch.float16)]
    out_storage = torch.full((2, 4, 130, 64), float("nan"), device="cuda", dtype=q.dtype)
    out = out_storage[:, ::2, ::2, :]
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    actual, actual_lse = launch_flash_fwd(q, k, v, out=out, lse=lse, causal=True, scale=0.2)
    expected, expected_lse = reference_attention(q, k, v, causal=True, scale=0.2)
    assert actual is out and actual_lse is lse
    torch.testing.assert_close(out, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-2, rtol=3e-2)
    assert torch.isnan(out_storage[:, 1::2]).all()
    assert torch.isnan(out_storage[:, ::2, 1::2]).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", nargs="+", type=int, default=[128, 257])
    parser.add_argument("--kv-len", type=int, default=None)
    parser.add_argument("--head-dim", type=int, choices=SUPPORTED_HEAD_DIMS, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print("| BxHxSQxSKxD | dtype | causal | tle_ms | max_abs | lse_max_abs | allclose |", flush=True)
    print("| --- | --- | --- | --- | --- | --- | --- |", flush=True)
    for seq_q in args.seq_len:
        seq_k = seq_q if args.kv_len is None else args.kv_len
        q, k, v = _make_inputs(args.batch, args.heads, seq_q, seq_k, args.head_dim, dtype)
        out, lse, errors = _check(q, k, v, args.causal)
        ms = "-"
        if args.benchmark:
            elapsed = triton.testing.do_bench(lambda: launch_flash_fwd(q, k, v, out=out, lse=lse, causal=args.causal),
                                              warmup=args.warmup, rep=args.rep)
            ms = f"{elapsed:.6f}"
        shape = f"{args.batch}x{args.heads}x{seq_q}x{seq_k}x{args.head_dim}"
        print(f"| {shape} | {args.dtype} | {args.causal} | {ms} | {errors[0]:.6g} | {errors[1]:.6g} | True |",
              flush=True)


if __name__ == "__main__":
    main()
