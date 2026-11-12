# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Retains the recurrent formulation from flash-linear-attention (MIT):
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""Standalone TLE recurrent GDN kernel.

The module has no framework-specific imports.  The only runtime requirements
are PyTorch, Triton, and Triton's MetaX TLE extension.
"""
import math

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _recurrent_gdn_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    State,
    Out,
    Final,
    Cu,
    Indices,
    Accepted,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    POOL: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    SS: tl.constexpr,
    IX: tl.constexpr,
    WIDTH: tl.constexpr,
    VARLEN: tl.constexpr,
    INDEXED: tl.constexpr,
    TABLE: tl.constexpr,
    SPEC: tl.constexpr,
    INITIAL: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = tl.cdiv(V, BV)
    iv = pid % nv
    nh = pid // nv
    n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    if VARLEN:
        start = tl.load(Cu + n).to(tl.int64)
        end = tl.load(Cu + n + 1).to(tl.int64)
        batch = 0
    else:
        start, end = 0, T
        batch = n
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    mask = (rv[:, None] < V) & (rk[None, :] < K)
    slot = n.to(tl.int64)
    valid = True
    if INDEXED:
        col = 0
        if SPEC:
            col = tl.load(Accepted + n).to(tl.int64) - 1
        valid = (col >= 0) & (col < WIDTH)
        slot = tl.load(Indices + n * IX[0] + col * IX[1], mask=valid, other=0).to(
            tl.int64
        )
        valid = valid & (slot > 0) & (slot < POOL)
    state_offsets = hv * SS[1] + rv[:, None] * SS[2] + rk[None, :] * SS[3]
    acc = tl.full((BV, BK), 0, tl.float32)
    if INITIAL and K == BK and V % BV == 0 and not INDEXED:
        state_tile = tle.gpu.alloc(
            [BV, BK], dtype=tl.float32, layout=None,
            scope=tle.gpu.smem, nv_mma_shared_layout=False,
        )
        tle.gpu.metax.async_copy_global_to_local(
            State + slot * SS[0] + state_offsets, state_tile,
        )
    elif INITIAL:
        acc = tl.load(
            State + slot * SS[0] + state_offsets, mask=mask & valid, other=0
        ).to(tl.float32)
    for t in range(start, end):
        q = tl.load(
            Q + batch * QS[0] + t * QS[1] + h * QS[2] + rk * QS[3], rk < K, 0
        ).to(tl.float32)
        k = tl.load(
            Kp + batch * KS[0] + t * KS[1] + h * KS[2] + rk * KS[3], rk < K, 0
        ).to(tl.float32)
        v = tl.load(
            Vp + batch * VS[0] + t * VS[1] + hv * VS[2] + rv * VS[3], rv < V, 0
        ).to(tl.float32)
        if NORMALIZE:
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
        q *= scale
        if INITIAL and K == BK and V % BV == 0 and not INDEXED:
            if t == start:
                tle.gpu.metax.gvm_arrive(0)
                acc = tl.load(tle.gpu.local_ptr(state_tile))
        if HAS_G:
            g = tl.load(G + batch * GS[0] + t * GS[1] + hv * GS[2]).to(tl.float32)
            acc *= tl.exp(g)
        delta = v - tl.sum(acc * k[None, :], 1)
        if HAS_BETA:
            bp = Beta + batch * BS[0] + t * BS[1] + hv * BS[2]
            if VECTOR_BETA:
                beta = tl.load(bp + rv * BS[3], rv < V, 0).to(tl.float32)
            else:
                beta = tl.load(bp).to(tl.float32)
            delta *= beta
        acc += delta[:, None] * k[None, :]
        out = tl.sum(acc * q[None, :], 1)
        token = batch * T + t
        tl.store(Out + (token * HV + hv) * V + rv, tl.where(valid, out, 0), rv < V)
        if INPLACE:
            dest = slot
            store = valid
            if TABLE:
                final_col = t - start
                dest = tl.load(
                    Indices + n * IX[0] + final_col * IX[1], final_col < WIDTH, 0
                ).to(tl.int64)
                store = store & (dest > 0) & (dest < POOL) & (final_col < WIDTH)
            else:
                store = store & (t == end - 1)
            tl.store(Final + dest * SS[0] + state_offsets, acc, mask & store)
        else:
            offsets = ((token * HV + hv) * V + rv[:, None]) * K + rk[None, :]
            tl.store(Final + offsets, tl.where(valid, acc, 0), mask)


def fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Return output and state in [pool, HV, V, K] layout.

    A 1D index maps each sequence to one cache slot (last-state writeback).
    A 2D index table stores each token's intermediate state, as required by
    speculative decoding; accepted counts select the initial table column.
    Initial slots <=0 or >=pool and invalid accepted columns produce zeros
    without updating the cache. Invalid destination slots suppress that write.
    Writable slots across sequences must be disjoint.
    Without indices, cache slots are sequence-major. Out-of-place mode returns
    every token state as [B*T, HV, V, K] and never modifies initial_state.
    cu_seqlens must be monotone, start at zero and end at T. Index-table width
    must cover each sequence. These device metadata are caller preconditions.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k must have identical [B,T,H,K] shapes; v must be rank 4")
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    if v.shape[:2] != (B, T) or min(H, HV, K, V) <= 0 or HV % H:
        raise ValueError("Invalid GVA dimensions")
    if K > 256:
        raise NotImplementedError("Recurrent GDN supports K <= 256")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("Recurrent GDN supports FP16, BF16 and FP32")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must have the same dtype")
    if scale is None:
        scale = K**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1:
            raise ValueError("Packed varlen requires B=1 and cu_seqlens[N+1]")
        if not cu_seqlens.is_contiguous() or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cu_seqlens must be contiguous int32/int64")
        N = cu_seqlens.numel() - 1
    else:
        N = B
    if inplace_final_state and initial_state is None:
        raise ValueError("In-place mode requires initial_state")
    if initial_state is not None:
        if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, V, K):
            raise ValueError("initial_state must use [pool, HV, V, K] layout")
        if initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Unsupported state dtype")
        if any(s <= 0 for s in initial_state.stride()):
            raise ValueError("State must have positive non-overlapping strides")
        ordered = sorted(zip(initial_state.stride(), initial_state.shape))
        extent = 1
        for stride, size in ordered:
            if size > 1 and stride < extent:
                raise ValueError("Overlapping state is unsupported")
            extent = max(extent, stride * size)
        pool, ss = initial_state.shape[0], initial_state.stride()
        if ssm_state_indices is None and pool < N:
            raise ValueError("State pool is smaller than the sequence count")
    else:
        pool, ss = N, (HV * V * K, V * K, K, 1)
    ix, width = (0, 0), 1
    if ssm_state_indices is not None:
        if initial_state is None:
            raise ValueError("Indexed mode requires an initial state pool")
        if ssm_state_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("State indices must be int32/int64")
        if ssm_state_indices.ndim not in (1, 2) or ssm_state_indices.shape[0] != N:
            raise ValueError("State indices must have shape [N] or [N,width]")
        if ssm_state_indices.ndim == 2:
            width = ssm_state_indices.shape[1]
            if width == 0:
                raise ValueError("State table cannot have zero width")
            ix = ssm_state_indices.stride()
        else:
            ix = (ssm_state_indices.stride(0), 0)
    if num_accepted_tokens is not None:
        if ssm_state_indices is None or ssm_state_indices.ndim != 2:
            raise ValueError("Accepted-token counts require a 2D state table")
        if (
            num_accepted_tokens.shape != (N,)
            or not num_accepted_tokens.is_contiguous()
            or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Accepted-token counts must be contiguous int32/int64 [N]")
    if g is not None and g.shape != (B, T, HV):
        raise ValueError("g must have shape [B,T,HV]")
    if beta is not None and beta.shape not in ((B, T, HV), (B, T, HV, V)):
        raise ValueError("beta must have shape [B,T,HV] or [B,T,HV,V]")
    for tensor in (
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on the same device")
    for tensor in (g, beta):
        if tensor is not None and tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("g/beta must be floating point")
    o = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty(
            (B * T, HV, V, K),
            dtype=initial_state.dtype if initial_state is not None else torch.float32,
            device=q.device,
        )
    )
    if B * T == 0:
        return o, final
    if N == 0:
        raise ValueError("Nonempty input requires at least one sequence")
    # No-autotune exemption: replaying this kernel mutates its state pool.
    # Tile experiments must restore state outside timing, never in production.
    bv = 8
    bs = beta.stride() if beta is not None else (0, 0, 0)
    if len(bs) == 3:
        bs = (*bs, 0)
    total = N * HV * triton.cdiv(V, bv)
    grid, kernel = (total,), _recurrent_gdn_kernel
    kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        o,
        final,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        T,
        H,
        HV,
        K,
        V,
        pool,
        q.stride(),
        k.stride(),
        v.stride(),
        g.stride() if g is not None else (0, 0, 0),
        bs,
        ss,
        ix,
        width,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        ssm_state_indices is not None and ssm_state_indices.ndim == 2,
        num_accepted_tokens is not None,
        initial_state is not None,
        inplace_final_state,
        beta is not None and beta.ndim == 4,
        use_qk_l2norm_in_kernel,
        g is not None,
        beta is not None,
        triton.next_power_of_2(K),
        bv,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return o, final


def benchmark_fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    *,
    warmup=25,
    rep=100,
    inplace_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Return the median latency in milliseconds using ``do_bench``.

    The timed callable uses out-of-place final-state mode by default, so
    repeated samples do not carry state updates from one iteration into the
    next.  A caller can pass any tensors accepted by
    :func:`fused_recurrent_gated_delta_rule_fwd`; no reference implementation
    or external package is imported here.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    def run():
        return fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            inplace_final_state=inplace_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    # ``device_type=metax`` is understood by the MetaX Triton test helper.
    return triton.testing.do_bench(
        run,
        warmup=warmup,
        rep=rep,
        return_mode="median",
        device_type="metax",
    )


__all__ = [
    "fused_recurrent_gated_delta_rule_fwd",
    "benchmark_fused_recurrent_gated_delta_rule",
]


# ---------------------------------------------------------------------------
# Embedded mcoplib MACA reference implementation.
# This section is copied into the submission file so runtime users do not need
# to install or import mcoplib.
# ---------------------------------------------------------------------------

# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from functools import wraps


def input_guard(fn):
    """Require dense tensors for legacy recurrent paths; never copy state pools.

    This local guard deliberately rejects unsupported layouts instead of silently
    cloning mutable state. Packed decode has its own stride-aware validation.
    """
    @wraps(fn)
    def guarded(*args, **kwargs):
        tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
        if tensors:
            device = tensors[0].device
            if device.type != "cuda" or any(t.device != device for t in tensors):
                raise ValueError("All tensors must be on the same GPU")
            if any(not t.is_contiguous() for t in tensors):
                raise ValueError("Recurrent forward/update require contiguous tensors")
            with torch.cuda.device(device):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return guarded


# Forward-only port of the supplied fused_recurrent_rzc_final.py.
# The original SGLang exp helper was not supplied. tl.exp matches the standalone
# replay implementation; numerical equivalence must be checked on target hardware.
__all__ = [
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "fused_recurrent_gated_delta_rule_update",
    "fused_recurrent_gdn",
]


@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * H + i_h) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h *= tl.exp(b_gk[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += H * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def mcoplib_fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
    else:
        final_state = None

    grid = (NK, NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state


# Adapted from vllm project.
@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    LOAD_ORDER: tl.constexpr = 0,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    # Experimental source schedules; the backend may still reorder instructions.
    if LOAD_ORDER == 0 or LOAD_ORDER == 2:
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    if LOAD_ORDER < 2:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        if LOAD_ORDER < 2:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    if LOAD_ORDER < 2:
        b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    if LOAD_ORDER == 1 or LOAD_ORDER == 3:
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    if LOAD_ORDER >= 2:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit
def fused_recurrent_gated_delta_rule_packed_decode_grouped_hv_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    REUSE_Q: tl.constexpr,
    GROUPED_SCHEDULE: tl.constexpr,
):
    """Process all value heads sharing one Q/K head in one program.

    The stable kernel launches one program per value head and therefore repeats
    the K load and K reduction HV/H times.  Qwen3.5-27B has HV/H=3.  This
    try schedule keeps normalized K live, processes those three value
    heads sequentially, and optionally keeps normalized Q live as well.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32)
        for i_g in tl.static_range(0, GROUP_SIZE):
            i_hv = i_h * GROUP_SIZE + i_g
            p_o = o + (i_n * HV + i_hv) * V + o_v
            tl.store(p_o, zero, mask=mask_v)
        return

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = H * K + i_h * K + o_k
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

    if REUSE_Q:
        b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_q = b_q * scale

    for i_g in tl.static_range(0, GROUP_SIZE):
        i_hv = i_h * GROUP_SIZE + i_g
        v_off = 2 * H * K + i_hv * V + o_v
        # Experimental schedules: bit 0 loads state before gate computation;
        # bit 1 writes updated state before output reduction. Default stays 0.
        if GROUPED_SCHEDULE & 1:
            p_h = h0 + state_idx * stride_init_state_token
            p_h = p_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        A_log_val = tl.load(A_log + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

        if not (GROUPED_SCHEDULE & 1):
            p_h = h0 + state_idx * stride_init_state_token
            p_h = p_h + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
        b_h *= tl.exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]

        if GROUPED_SCHEDULE & 2:
            p_ht = ht + state_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        if not REUSE_Q:
            b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_q = b_q * scale
        b_o = tl.sum(b_h * b_q[None, :], 1)
        p_o = o + (i_n * HV + i_hv) * V + o_v
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if not (GROUPED_SCHEDULE & 2):
            p_ht = ht + state_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    num_warps: int = 1,
    num_stages: int = 3,
    bv: Optional[int] = None,
    load_order: Optional[int] = None,
    group_value_heads: Optional[bool] = None,
    reuse_q: bool = False,
    grouped_schedule: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # None: select a measured C600 configuration; 0/1/2/3 force an experiment.
    # 0: original; 1: late state; 2: late Q; 3: late state and Q.
    if load_order is not None and load_order not in (0, 1, 2, 3):
        raise ValueError("load_order must be None, 0, 1, 2, or 3")
    if grouped_schedule not in (0, 1, 2, 3):
        raise ValueError("grouped_schedule must be 0, 1, 2, or 3")
    if grouped_schedule and group_value_heads is not True:
        raise ValueError("experimental grouped_schedule requires group_value_heads=True")
    if reuse_q and group_value_heads is False:
        raise ValueError("reuse_q requires group_value_heads=True")
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    if initial_state.ndim == 4 and (
        initial_state.stride(1) != initial_state.shape[2] * initial_state.shape[3]
        or initial_state.stride(2) != initial_state.shape[3]
    ):
        raise ValueError("State head/V/K dimensions must be dense; slot stride may vary")
    dev = mixed_qkv.device
    if dev.type != "cuda":
        raise ValueError("Packed decode requires GPU tensors")
    if any(
        t.device != dev
        for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )
    if group_value_heads and HV == H:
        raise ValueError(
            "group_value_heads requires more than one value head per Q/K head"
        )

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(
            f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK})."
        )
    BV = min(triton.next_power_of_2(V), 32) if bv is None else bv
    if BV <= 0 or BV & (BV - 1):
        raise ValueError("bv must be a positive power of two")
    if num_warps not in (1, 2, 4, 8) or num_stages < 1:
        raise ValueError("Unsupported launch configuration")

    if load_order is None:
        is_measured_shape = (
            ((H == 16 and HV == 32) or (H == 8 and HV == 24))
            and K == 128 and V == 128 and BV == 32 and num_warps == 1
        )
        load_order = 1 if is_measured_shape and B * HV <= 128 else (
            3 if is_measured_shape else 0
        )

    grouped_num_warps = num_warps
    if group_value_heads is None:
        is_measured_27b_tp2_shape = (
            H == 8 and HV == 24 and K == 128 and V == 128 and BV == 32
        )
        if is_measured_27b_tp2_shape and (
            B == 8 or (16 <= B <= 128 and B % 8 == 0)
        ):
            group_value_heads = True
            reuse_q = True
            BV = 16
            grouped_num_warps = 1
        elif is_measured_27b_tp2_shape and 4 <= B <= 8:
            group_value_heads = True
            reuse_q = True
            grouped_num_warps = 4
        elif is_measured_27b_tp2_shape and 16 <= B <= 128:
            group_value_heads = True
            reuse_q = True
            grouped_num_warps = 2
        else:
            group_value_heads = False

    if reuse_q and not group_value_heads:
        raise ValueError("reuse_q requires group_value_heads=True")

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    if group_value_heads:
        grid = (NV, B * H)
        fused_recurrent_gated_delta_rule_packed_decode_grouped_hv_kernel[grid](
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            o=out,
            h0=initial_state,
            ht=initial_state,
            ssm_state_indices=ssm_state_indices,
            scale=scale,
            stride_mixed_qkv_tok=stride_mixed_qkv_tok,
            stride_a_tok=stride_a_tok,
            stride_b_tok=stride_b_tok,
            stride_init_state_token=stride_init_state_token,
            stride_final_state_token=stride_final_state_token,
            stride_indices_seq=stride_indices_seq,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            GROUP_SIZE=HV // H,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            REUSE_Q=reuse_q,
            GROUPED_SCHEDULE=grouped_schedule,
            num_warps=grouped_num_warps,
            num_stages=num_stages,
        )
        return out, initial_state

    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        LOAD_ORDER=load_order,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, V, K]` if `output_final_state=True` else `None`.
    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, V, K, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state


# HAS_EAGLE_TREE_CUSTOM_ATTN_MASK is added to support eagle tree attention mask
# retrieve_parent_token_ptr: [N, NP2_T], retrieve_next_sibling_ptr: [N, NP2_T]
# e.g. for a sequence of length 4, the eagle tree attention structure is:
# retrieve_next_token=[1, 3, -1, -1] -> retrieve_next_token[i]: the 1st child token of token i
# retrieve_next_sibling=[-1, 2, -1, -1] -> retrieve_next_sibling[i]: the 1st tree sibling token of token i
# retrieve_parent_token=[n/a, 0, 0, 1] -> retrieve_parent_token[i]: the parent token of token i
# Tree:
#    0
#   / \
#  1   2
# /
# 3
# When calculating token 3's attention, it should attend to token 1 (parent) and token 0 (grand-parent)
# When calculating token 2's attention, it should attend to token 0 (parent)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_update_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    T,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DISABLE_STATE_UPDATE: tl.constexpr,  # whether to disable final state update
    DISABLE_OUTPUT_CALCULATION: tl.constexpr,  # whether to disable output calculation
    CACHE_INTERMEDIATE_STATES: tl.constexpr,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(retrieve_parent_token_base, mask_retrieve)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        # Add bounds checking for idx
        if idx >= 0:  # Assuming negative indices are invalid
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Prepare intermediate state cache variables if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n).to(tl.int64)

    step_idx = 0
    for _ in range(0, T):
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx = 0 should use the b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                # when calculating current step's attention, load the state from the parent token
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BK, BV]
        b_h *= tl.exp(b_g)
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        if not DISABLE_OUTPUT_CALCULATION:
            b_o = tl.sum(b_h * b_q[None, :], 1)
            # core attn output
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # store intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                # Compute cache pointer for this step
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[:, None] * K
                    + o_k[None, :]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    # Store final state back to h0_source with bounds checking
    # ssm states
    if not DISABLE_STATE_UPDATE and USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n).to(tl.int64)
        if idx >= 0:  # Add bounds checking
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[:, None] * K
                + o_k[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_update_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if initial_state_source is not None and initial_state_indices is None:
        raise ValueError("initial_state_indices is required with a state pool")
    if intermediate_states_buffer is not None:
        if intermediate_state_indices is None:
            raise ValueError("intermediate_state_indices is required with a cache")
        allocated_steps = intermediate_states_buffer.shape[1]
        if cache_steps is not None and cache_steps != allocated_steps:
            raise ValueError("cache_steps must match the allocated cache step dimension")
        cache_steps = allocated_steps
    if retrieve_parent_token is not None and intermediate_states_buffer is None:
        raise ValueError("Tree verify requires an intermediate state cache")
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if disable_output_calculation:
        # When output calculation is disabled, allocate minimal tensor
        o = q.new_empty(NK, 1, 1, 1, 1)  # minimal allocation
    else:
        o = q.new_empty(NK, *v.shape)

    grid = (NK, NV, N * HV)

    # prepare retrieve next token buffer strides if provided
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq, stride_retrieve_parent_token_token = (
            retrieve_parent_token.stride(0),
            retrieve_parent_token.stride(1),
        )
    else:
        stride_retrieve_parent_token_seq = stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)
    fused_recurrent_gated_delta_rule_update_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=0 if cache_steps is None else cache_steps,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        T=T,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        IS_VARLEN=cu_seqlens is not None,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        DISABLE_STATE_UPDATE=disable_state_update,
        DISABLE_OUTPUT_CALCULATION=disable_output_calculation,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


class FusedRecurrentUpdateFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        disable_state_update: bool = False,
        disable_output_calculation: bool = False,
        intermediate_states_buffer: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        cache_steps: Optional[int] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
    ):
        o = fused_recurrent_gated_delta_rule_update_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            disable_state_update=disable_state_update,
            disable_output_calculation=disable_output_calculation,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
        )

        return o

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )


def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state_source: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state_source is not None:
            if initial_state_indices.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
                )
            if intermediate_state_indices is not None and initial_state_indices.shape[0] != intermediate_state_indices.shape[0]:
                raise ValueError(
                    f"The number of intermediate state indices is expected to be equal to the number of input sequences, "
                    f"i.e., {initial_state_indices.shape[0]} != {intermediate_state_indices.shape[0]}."
                )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    o = FusedRecurrentUpdateFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        disable_state_update,
        disable_output_calculation,
        intermediate_states_buffer,
        intermediate_state_indices,
        cache_steps,
        retrieve_parent_token,
    )
    return o


fused_recurrent_gdn = fused_recurrent_gated_delta_rule


# ---------------------------------------------------------------------------
# Single-file correctness and performance helpers.
# ---------------------------------------------------------------------------

def benchmark_tle_vs_mcoplib(q, k, v, g, beta, scale, initial_state, *,
                             warmup=25, rep=100):
    """Return ``(tle_ms, maca_ms, maca_over_tle)`` using ``do_bench``.

    Both implementations receive the same tensors.  TLE runs out-of-place;
    the embedded MACA reference receives a cloned state because it updates its
    state argument in place.
    """
    import triton.testing

    def run_tle():
        return fused_recurrent_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, initial_state,
            inplace_final_state=False, use_qk_l2norm_in_kernel=True,
        )

    def run_maca():
        return mcoplib_fused_recurrent_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, initial_state.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )

    tle_ms = triton.testing.do_bench(
        run_tle, warmup=warmup, rep=rep, return_mode="median",
        device_type="metax",
    )
    maca_ms = triton.testing.do_bench(
        run_maca, warmup=warmup, rep=rep, return_mode="median",
        device_type="metax",
    )
    return tle_ms, maca_ms, maca_ms / tle_ms


__all__ = [
    "fused_recurrent_gated_delta_rule_fwd",
    "mcoplib_fused_recurrent_gated_delta_rule_fwd",
    "benchmark_tle_vs_mcoplib",
]


def _benchmark_inputs(batch, dtype):
    device = "cuda:0"
    T, H, HV, K, V = 1, 4, 8, 128, 128
    q = torch.randn((batch, T, H, K), device=device, dtype=dtype)
    k = torch.randn_like(q) * 0.05
    v = torch.randn((batch, T, HV, V), device=device, dtype=dtype)
    g = -torch.rand((batch, T, HV), device=device, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn((batch, T, HV), device=device, dtype=dtype))
    state = torch.randn((batch, HV, V, K), device=device, dtype=torch.float32) * 0.05
    return q, k, v, g, beta, K**-0.5, state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TLE vs embedded MACA GDN benchmark")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    print(f"{'B':>4} {'T':>3} {'H':>3} {'HV':>3} {'K':>4} {'V':>4} {'dtype':>8} {'tle_ms':>10} {'maca_ms':>10} {'maca/tle':>10}")
    for batch in args.batches:
        for dtype in (torch.float16, torch.bfloat16):
            q, k, v, g, beta, scale, state = _benchmark_inputs(batch, dtype)
            tle_ms, maca_ms, speedup = benchmark_tle_vs_mcoplib(
                q, k, v, g, beta, scale, state,
                warmup=args.warmup, rep=args.rep,
            )
            print(
                f"{batch:4d} {1:3d} {4:3d} {8:3d} {128:4d} {128:4d} "
                f"{str(dtype).split('.')[-1]:>8} {tle_ms:10.6f} "
                f"{maca_ms:10.6f} {speedup:10.6f}",
                flush=True,
            )

