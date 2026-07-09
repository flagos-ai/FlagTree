# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
import inspect
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import get_available_triton_backend
from tilegym.backend import mark_perf_ready
from tilegym.backend import register_impl
from tilegym.logger import get_logger

# Constants & Type Aliases

logger = get_logger(__name__)

INV_LOG_2 = tl.constexpr(1.0 / math.log(2))
# Check if triton.Config supports v2_opt_level parameter

# Capability Probe


def _supports_v2_opt_level():
    try:
        sig = inspect.signature(triton.Config.__init__)
        return "v2_opt_level" in sig.parameters
    except Exception:
        return False


_TRITON_SUPPORTS_V2_OPT_LEVEL = _supports_v2_opt_level()


def _supports_host_descriptor():
    return torch.cuda.get_device_capability()[0] >= 9


def _supports_tma(tensor: torch.Tensor):
    # Check if the tensor stride is divisible by 16 bytes
    # Mainly for the non-even sequence length case
    return torch.finfo(tensor.dtype).bits * tensor.stride(-2) // 8 % 16 == 0


# Kernel Helpers


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd_inner(
    K_desc,
    V_desc,
    acc,
    l_i,
    m_i,
    q,
    batch_idx,
    off_kv_h,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    EVEN_K: tl.constexpr,
    warp_specialize: tl.constexpr,
    prefix_kvlen,
):
    # Range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M  # start_m=0,1,2,3
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    cnt = lo // BLOCK_N
    # Loop over k, v and update accumulator
    # Here use warp_specialize=True only for best performance in oait, for nvt, it will ignore this argument
    for curr_n in range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        curr_n = tl.multiple_of(curr_n, BLOCK_N)
        # -- Compute qk ----
        k = K_desc.load([batch_idx, off_kv_h, cnt * BLOCK_N, 0])
        k = tl.reshape(k, (BLOCK_N, BLOCK_D))
        # Transpose K to get (BLOCK_D, BLOCK_N) for matrix multiplication
        k = tl.trans(k)
        qk = tl.dot(q, k)
        # Process boundary case(here we only need to process non-causal case)
        if STAGE == 3 and not EVEN_K:
            mask = curr_n + offs_n[None, :] < N_CTX
            qk = tl.where(mask, qk, -1.0e6)
        # Apply causal mask: query at global position (prefix_kvlen + offs_m) attends to KV[0..that pos]
        if STAGE == 2:
            mask = (prefix_kvlen + offs_m[:, None]) >= (curr_n + offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        # Attention weights
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # Update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # Update output accumulator
        acc = acc * alpha[:, None]
        # Update acc
        v = V_desc.load([batch_idx, off_kv_h, cnt * BLOCK_N, 0])
        v = tl.reshape(v, (BLOCK_N, BLOCK_D))
        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        cnt += 1

    return acc, l_i, m_i


# The `_impl` suffix denotes the core kernel logic, separated from the
# @triton.autotune'd entry point so it can be reused by different kernels
# (e.g. standalone prefill attention and fused POD attention) without
# duplicating the attention computation code.
@triton.jit
def prefill_fmha_impl(
    Q,
    K,
    V,
    Out,
    L,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    S_kv,
    BHS_qo,
    BHS_kv,
    pid_x,
    pid_y,
    HAS_BACKWARD: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D = hidden_size
    STAGE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    dtype: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    if isinstance(Q, tl.tensor_descriptor):
        dtype = Q.type.block_type.element_ty
    else:
        dtype = Q.dtype.element_ty
    batch_idx = pid_y // H
    head_idx = pid_y % H
    if QUERY_GROUP_SIZE:
        off_kv_h = head_idx // QUERY_GROUP_SIZE
    else:
        off_kv_h = head_idx
    qk_scale = sm_scale * INV_LOG_2

    # Initialize offsets
    offs_m = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Create tensor descriptors
    Q_desc = _maybe_make_tensor_desc(
        Q,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    # For K, we need to load BLOCK_N x BLOCK_D but then transpose to BLOCK_D x BLOCK_N
    K_desc = _maybe_make_tensor_desc(
        K,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )

    V_desc = _maybe_make_tensor_desc(
        V,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )

    O_desc = _maybe_make_tensor_desc(
        Out,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    if USE_DESC_FOR_CORR:
        L_desc = _maybe_make_tensor_desc(
            L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )

    # Initialize m, l, acc
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q - calculate explicit offsets
    q_offset_b = batch_idx * 1
    q_offset_h = head_idx * 1
    q_offset_m = pid_x * BLOCK_M
    q_offset_d = 0 * BLOCK_D
    q = Q_desc.load([q_offset_b, q_offset_h, q_offset_m, q_offset_d])
    q = tl.reshape(q, (BLOCK_M, BLOCK_D))

    # chunked prefill: query token i attends to KV[0..prefix_kvlen+i]
    prefix_kvlen = S_kv - S_qo
    start_m = prefix_kvlen // BLOCK_M + pid_x

    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            K_desc,
            V_desc,
            acc,
            l_i,
            m_i,
            q,
            batch_idx,
            off_kv_h,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            4 - STAGE,
            offs_m,
            offs_n,
            S_kv,
            EVEN_K,
            warp_specialize,
            prefix_kvlen,
        )
    # Stage 2: on-band
    if STAGE & 2:
        # Barrier makes it easier for compielr to schedule the
        # Two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            K_desc,
            V_desc,
            acc,
            l_i,
            m_i,
            q,
            batch_idx,
            off_kv_h,
            start_m,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            2,
            offs_m,
            offs_n,
            S_kv,
            EVEN_K,
            warp_specialize,
            prefix_kvlen,
        )
    # Epilogue
    acc = acc / (l_i[:, None])
    l_i = m_i + tl.math.log2(l_i)

    # Write back l and o - calculate explicit offsets
    if HAS_BACKWARD:
        l_offset_b = batch_idx * 1
        l_offset_h = head_idx * 1
        l_offset_m = pid_x * BLOCK_M
        if USE_DESC_FOR_CORR:
            L_desc.store([l_offset_b, l_offset_h, l_offset_m], l_i.reshape(1, 1, BLOCK_M))
        else:
            offset_s_ptrs = l_offset_m + tl.reshape(tl.arange(0, BLOCK_M), (1, 1, BLOCK_M))
            offset_ptrs = l_offset_b * stride_lb + l_offset_h * stride_lh + offset_s_ptrs
            tl.store(L + offset_ptrs, l_i.reshape(1, 1, BLOCK_M), mask=offset_s_ptrs < S_qo)

    O_desc.store(
        [batch_idx, head_idx, pid_x * BLOCK_M, 0],
        acc.to(dtype).reshape(1, 1, BLOCK_M, BLOCK_D),
    )


# Autotune Config


def _host_descriptor_pre_hook_fwd(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    if not isinstance(nargs["Q"], TensorDescriptor):
        return
    nargs["Q"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
    nargs["K"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
    nargs["V"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
    nargs["Out"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
    if isinstance(nargs["L"], TensorDescriptor):
        nargs["L"].block_shape = [1, 1, BLOCK_M]


def _host_descriptor_pre_hook_bwd(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    if not isinstance(nargs["dO"], TensorDescriptor):
        return
    nargs["dO"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
    # bwd preprocess kernel
    if "minus_L" in nargs:
        nargs["Out"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
        if isinstance(nargs["minus_L"], TensorDescriptor):
            nargs["minus_L"].block_shape = [1, 1, BLOCK_M]
            nargs["minus_Delta"].block_shape = [1, 1, BLOCK_M]
            nargs["L"].block_shape = [1, 1, BLOCK_M]
    # bwd kernel
    else:
        nargs["Q"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
        nargs["K"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
        nargs["V"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
        nargs["dQ"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
        nargs["dK"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
        nargs["dV"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
        if isinstance(nargs["L"], TensorDescriptor):
            nargs["L"].block_shape = [1, 1, BLOCK_M]
            nargs["Delta"].block_shape = [1, 1, BLOCK_M]


def _get_configs(is_backward=False):
    if _supports_host_descriptor():
        _hook = _host_descriptor_pre_hook_bwd if is_backward else _host_descriptor_pre_hook_fwd
    else:
        _hook = None

    if get_available_triton_backend() == "nvt":
        default_config = {"warp_specialize": False}
        if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
            configs = [
                triton.Config(dict(BLOCK_M=64, BLOCK_N=64, occupancy=2, **default_config), pre_hook=_hook),
            ]
        elif torch.cuda.get_device_capability() == (9, 0):
            configs = [
                # occupancy = 2. We currently do not expose num_warps in CudaTile.
                # Once we do (and if we do), we can experiment with occupancy = 1 and num_warps = 8.
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, **default_config), num_stages=s, pre_hook=_hook)
                for BM in [64, 128]
                for BN in [64, 128]
                for s in [2, 3]
            ]
        elif torch.cuda.get_device_capability() == (8, 0):
            configs = [
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=occ, **default_config), num_stages=s,
                              pre_hook=_hook)
                for BM in [64, 128]
                for BN in [32, 64]
                for s in [2, 3, 4]
                for occ in [1, 2]
            ]
        else:
            if is_backward:
                configs = [
                    triton.Config(dict(BLOCK_M=128, BLOCK_N=128, **default_config), num_stages=7, pre_hook=_hook)
                ]
            else:
                configs = [
                    triton.Config(dict(BLOCK_M=256, BLOCK_N=128, **default_config), pre_hook=_hook),
                    triton.Config(dict(BLOCK_M=128, BLOCK_N=128, occupancy=2, **default_config), pre_hook=_hook),
                    triton.Config(dict(BLOCK_M=256, BLOCK_N=128, occupancy=2, **default_config), pre_hook=_hook),
                ]
                # GB300 (sm103): avoid num_ctas>1 which causes PTX codegen error
                # (mixed cta_group::1 and cta_group::2). See CFK-33703.
                if torch.cuda.get_device_capability() != (10, 3):
                    configs.append(
                        triton.Config(dict(BLOCK_M=256, BLOCK_N=128, occupancy=2, **default_config), num_ctas=2,
                                      pre_hook=_hook))
    elif get_available_triton_backend() == "oait":
        # Full tuning space for oait
        if is_backward:
            # Force num_stages=1 to work around oait bug for bwd, see https://github.com/triton-lang/triton/issues/7386
            configs = [
                triton.Config(dict(BLOCK_M=128, BLOCK_N=64, warp_specialize=False), num_stages=1, num_warps=4,
                              pre_hook=_hook)
            ]
            configs = list(configs)
        else:
            configs = [
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, warp_specialize=ws), num_stages=s, num_warps=w,
                              pre_hook=_hook)
                for BM in [64, 128, 256]
                for BN in [64, 128]
                for s in [2, 3, 4]
                for w in [4, 8]
                for ws in [True, False]
            ]
        if torch.cuda.get_device_capability() == (8, 0):
            configs = [
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, warp_specialize=False), num_stages=s, num_warps=w)
                for BM in [64, 128]
                for BN in [32, 64]
                for s in [2, 4]
                for w in [4, 8]
            ]
    return configs


def _prune_invalid_configs(configs, named_args, **kwargs):
    # Filter out configs where BLOCK_M % BLOCK_N != 0 when causal is True
    if kwargs["STAGE"] == 3:
        return [conf for conf in configs if conf.kwargs["BLOCK_M"] % conf.kwargs["BLOCK_N"] == 0]
    else:
        return configs


# Device Kernels


@triton.autotune(
    configs=_get_configs(),
    key=["S_qo", "S_kv", "BLOCK_D", "STAGE", "QUERY_GROUP_SIZE", "dtype"],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.heuristics({
    "EVEN_K": lambda args: args["S_kv"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _prefill_fmha(
    Q,
    K,
    V,
    Out,
    L,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    S_kv,
    BHS_qo,
    BHS_kv,
    HAS_BACKWARD: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D = hidden_size
    STAGE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    dtype: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    prefill_fmha_impl(
        Q,
        K,
        V,
        Out,
        L,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_lb,
        stride_lh,
        B,
        H,
        S_qo,
        S_kv,
        BHS_qo,
        BHS_kv,
        pid_x,
        pid_y,
        HAS_BACKWARD,
        USE_DESC_FOR_CORR,
        BLOCK_D,
        STAGE,
        QUERY_GROUP_SIZE,
        BLOCK_M,
        BLOCK_N,
        EVEN_K,
        dtype,
        warp_specialize,
    )


@triton.autotune(
    configs=_get_configs(is_backward=True),
    key=["S_qo", "USE_DESC_FOR_CORR"],
)
@triton.jit
def _fmha_bwd_preprocess_kernel(
    Out,
    dO,
    L,
    minus_Delta,
    minus_L,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    O_desc = _maybe_make_tensor_desc(
        Out,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dO_desc = _maybe_make_tensor_desc(
        dO,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    if USE_DESC_FOR_CORR:
        L_desc = _maybe_make_tensor_desc(
            L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        minus_Delta_desc = _maybe_make_tensor_desc(
            minus_Delta,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        minus_L_desc = _maybe_make_tensor_desc(
            minus_L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
    offset_b = pid_y // H
    offset_h = pid_y % H
    offset_s = (pid_x * BLOCK_M).to(tl.int32)
    o = O_desc.load([offset_b, offset_h, offset_s, 0]).to(tl.float32)
    do = dO_desc.load([offset_b, offset_h, offset_s, 0]).to(tl.float32)
    offset_s_ptrs = offset_s + tl.reshape(tl.arange(0, BLOCK_M), (1, 1, BLOCK_M))
    offset_ptrs = offset_b * stride_lb + offset_h * stride_lh + offset_s_ptrs
    if USE_DESC_FOR_CORR:
        l = L_desc.load([offset_b, offset_h, offset_s])
    else:
        l = tl.load(L + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)

    delta = -tl.sum(o * do, axis=3) * softmax_scale
    ml = -l
    if USE_DESC_FOR_CORR:
        minus_Delta_desc.store([offset_b, offset_h, offset_s], delta)
        minus_L_desc.store([offset_b, offset_h, offset_s], ml)
    else:
        tl.store(minus_Delta + offset_ptrs, delta, mask=offset_s_ptrs < S_qo)
        tl.store(minus_L + offset_ptrs, ml, mask=offset_s_ptrs < S_qo)


# Note: only ensure the functional correctness, not performance for now
@triton.autotune(
    configs=_get_configs(is_backward=True),
    key=["S_qo", "S_kv", "USE_DESC_FOR_CORR", "IS_CAUSAL"],
    reset_to_zero=["dq_ptr"],
)
@triton.jit
def _fmha_bwd_kernel(
    Q,
    K,
    V,
    dO,
    L,
    Delta,
    dQ,
    dq_ptr,  # used to reset dQ to zero during autotuning
    dK,
    dV,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    B,
    H,
    S_qo,
    S_kv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_DESC_FOR_CORR: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    if isinstance(Q, tl.tensor_descriptor):
        dtype = Q.type.block_type.element_ty
    else:
        dtype = Q.dtype.element_ty

    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if not IS_CAUSAL:
        start_m = 0
    else:
        start_m = ((pid_x * BLOCK_N) // BLOCK_M) * BLOCK_M

    Q_desc = _maybe_make_tensor_desc(
        Q,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    K_desc = _maybe_make_tensor_desc(
        K,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    V_desc = _maybe_make_tensor_desc(
        V,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    dO_desc = _maybe_make_tensor_desc(
        dO,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_ob, stride_oh, stride_om, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dQ_desc = _maybe_make_tensor_desc(
        dQ,
        shape=[B, H, S_qo, BLOCK_D],
        strides=[stride_qb, stride_qh, stride_qm, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    dK_desc = _maybe_make_tensor_desc(
        dK,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_kb, stride_kh, stride_kn, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    dV_desc = _maybe_make_tensor_desc(
        dV,
        shape=[B, H, S_kv, BLOCK_D],
        strides=[stride_vb, stride_vh, stride_vk, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_D],
    )
    if USE_DESC_FOR_CORR:
        L_desc = _maybe_make_tensor_desc(
            L,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )
        Delta_desc = _maybe_make_tensor_desc(
            Delta,
            shape=[B, H, S_qo],
            strides=[stride_lb, stride_lh, 1],
            block_shape=[1, 1, BLOCK_M],
        )

    offset_b = pid_y // H
    offset_h = pid_y % H
    offset_skv = pid_x * BLOCK_N

    offset_s_ptrs = start_m + tl.reshape(tl.arange(0, BLOCK_M), (1, 1, BLOCK_M))
    offset_ptrs = offset_b * stride_lb + offset_h * stride_lh + offset_s_ptrs

    k = K_desc.load([offset_b, offset_h, offset_skv, 0]).reshape(BLOCK_N, BLOCK_D)
    v = V_desc.load([offset_b, offset_h, offset_skv, 0]).reshape(BLOCK_N, BLOCK_D)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    softmax_scale_inv_ln2 = softmax_scale * tl.constexpr(1.0 / math.log(2))  # INV_LOG_2

    for curr_m in range(start_m, S_qo, BLOCK_M, warp_specialize):
        curr_m = tl.multiple_of(curr_m, BLOCK_M)
        q = Q_desc.load([offset_b, offset_h, curr_m, 0]).reshape(BLOCK_M, BLOCK_D)
        s_t = tl.dot(k, tl.trans(q))
        do = dO_desc.load([offset_b, offset_h, curr_m, 0]).reshape(BLOCK_M, BLOCK_D)
        dp_t = tl.dot(v, tl.trans(do))
        if USE_DESC_FOR_CORR:
            l = L_desc.load([offset_b, offset_h, curr_m])
        else:
            l = tl.load(L + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)
        if IS_CAUSAL:
            offs_n = offset_skv + tl.arange(0, BLOCK_N)
            offs_m_curr = curr_m + tl.arange(0, BLOCK_M)
            s_t = tl.where(offs_m_curr[None, :] >= offs_n[:, None], s_t, float("-inf"))
        if USE_DESC_FOR_CORR:
            delta = Delta_desc.load([offset_b, offset_h, curr_m])
        else:
            delta = tl.load(Delta + offset_ptrs, mask=offset_s_ptrs < S_qo, other=0.0)
        l_t = tl.view(l, (1, BLOCK_M))
        # replace sub l_t to add l_t to use fma2 instruction
        s_t = softmax_scale_inv_ln2 * s_t + l_t
        p_t = tl.exp2(s_t)
        p_t_f16 = p_t.to(dtype)
        delta_t = tl.view(delta, (1, BLOCK_M))
        dp_t_new = softmax_scale * dp_t + delta_t
        ds_t = (p_t * dp_t_new).to(dtype)
        dk = tl.dot(ds_t, q, dk)
        dv = tl.dot(p_t_f16, do, dv)
        ds = tl.trans(ds_t)
        dq = tl.dot(ds, k)
        dQ_desc.atomic_add(
            [offset_b, offset_h, curr_m, 0],
            dq.reshape(1, 1, BLOCK_M, BLOCK_D),
        )
        if not USE_DESC_FOR_CORR:
            offset_s_ptrs += BLOCK_M
            offset_ptrs += BLOCK_M
    dK_desc.store([offset_b, offset_h, offset_skv, 0], dk.to(dtype).reshape(1, 1, BLOCK_N, BLOCK_D))
    dV_desc.store([offset_b, offset_h, offset_skv, 0], dv.to(dtype).reshape(1, 1, BLOCK_N, BLOCK_D))


# Host Launchers & Public API


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        is_causal,
        has_backward=False,
    ):
        B, H, S_qo, BLOCK_D = q.shape
        assert k.shape == v.shape
        num_head_kv = k.shape[1]
        S_kv = k.shape[2]
        BHS_qo = B * H * S_qo
        BHS_kv = B * H * S_kv
        o = torch.empty_like(q)
        l = torch.empty((B, H, S_qo), device=q.device, dtype=torch.float32)
        stage = 3 if is_causal else 1
        if H == num_head_kv:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)

        # Launch fmha fwd kernel
        grid = lambda args: (triton.cdiv(S_qo, args["BLOCK_M"]), B * H, 1)
        USE_DESC_FOR_CORR = False

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        if _supports_host_descriptor():
            dummy_block_qo = [1, 1, 1, 1]
            dummy_block_kv = [1, 1, 1, 1]
            dummy_block_l = [1, 1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[q.stride(0), q.stride(1), q.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            desc_v = TensorDescriptor(
                v,
                shape=[B, num_head_kv, S_kv, BLOCK_D],
                strides=[v.stride(0), v.stride(1), v.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            desc_k = TensorDescriptor(
                k,
                shape=[B, num_head_kv, S_kv, BLOCK_D],
                strides=[k.stride(0), k.stride(1), k.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[o.stride(0), o.stride(1), o.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            if has_backward and _supports_tma(l):
                desc_l = TensorDescriptor(
                    l,
                    shape=[B, H, S_qo],
                    strides=[l.stride(0), l.stride(1), 1],
                    block_shape=dummy_block_l,
                )
                USE_DESC_FOR_CORR = True
            else:
                desc_l = l
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o
            desc_l = l

        # disable autotune when run test_op in test_attention.py
        if is_autotune_disabled():
            _prefill_fmha.configs = [_prefill_fmha.configs[0]]

        _prefill_fmha[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            desc_l,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            B,
            H,
            S_qo,
            S_kv,
            BHS_qo,
            BHS_kv,
            HAS_BACKWARD=has_backward,
            USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
            BLOCK_D=BLOCK_D,
            STAGE=stage,
            QUERY_GROUP_SIZE=query_group_size,
            dtype=q.dtype,
        )

        ctx.save_for_backward(q, k, v, o, l)
        ctx.sm_scale = sm_scale
        ctx.shapes = (B, H, S_qo, S_kv)
        ctx.launch_configs = (
            is_causal,
            BLOCK_D,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        B, H, S_qo, S_kv = ctx.shapes
        is_causal, BLOCK_D = ctx.launch_configs
        USE_DESC_FOR_CORR = False
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        minus_delta = torch.empty_like(l)
        minus_l = torch.empty_like(l)
        assert q.shape[1] == k.shape[1], "bwd for FMHAGQA is not supported for now."
        assert dq.stride() == q.stride()
        assert dk.stride() == k.stride()
        assert dv.stride() == v.stride()
        assert do.stride() == o.stride()

        if _supports_host_descriptor():
            dummy_block_qo = [1, 1, 1, 1]
            dummy_block_kv = [1, 1, 1, 1]
            dummy_block_ld = [1, 1, 1]
            Q_desc = TensorDescriptor(
                q,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[q.stride(0), q.stride(1), q.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            K_desc = TensorDescriptor(
                k,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[k.stride(0), k.stride(1), k.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            V_desc = TensorDescriptor(
                v,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[v.stride(0), v.stride(1), v.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            dO_desc = TensorDescriptor(
                do,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[o.stride(0), o.stride(1), o.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            O_desc = TensorDescriptor(
                o,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[o.stride(0), o.stride(1), o.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            dQ_desc = TensorDescriptor(
                dq,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[dq.stride(0), dq.stride(1), dq.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            dK_desc = TensorDescriptor(
                dk,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[dk.stride(0), dk.stride(1), dk.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            dV_desc = TensorDescriptor(
                dv,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[dv.stride(0), dv.stride(1), dv.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            if _supports_tma(l):
                L_desc = TensorDescriptor(
                    l,
                    shape=[B, H, S_qo],
                    strides=[l.stride(0), l.stride(1), 1],
                    block_shape=dummy_block_ld,
                )
                minus_L_desc = TensorDescriptor(
                    minus_l,
                    shape=[B, H, S_qo],
                    strides=[minus_l.stride(0), minus_l.stride(1), 1],
                    block_shape=dummy_block_ld,
                )
                Delta_desc = TensorDescriptor(
                    minus_delta,
                    shape=[B, H, S_qo],
                    strides=[minus_delta.stride(0), minus_delta.stride(1), 1],
                    block_shape=dummy_block_ld,
                )
                USE_DESC_FOR_CORR = True
            else:
                L_desc = l
                minus_L_desc = minus_l
                Delta_desc = minus_delta
        else:
            Q_desc = q
            K_desc = k
            V_desc = v
            dO_desc = do
            O_desc = o
            dQ_desc = dq
            dK_desc = dk
            dV_desc = dv
            L_desc = l
            minus_L_desc = minus_l
            Delta_desc = minus_delta

        grid = lambda args: (triton.cdiv(S_qo, args["BLOCK_M"]), B * H, 1)
        _fmha_bwd_preprocess_kernel[grid](
            O_desc,
            dO_desc,
            L_desc,
            Delta_desc,
            minus_L_desc,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            B,
            H,
            S_qo,
            ctx.sm_scale,
            BLOCK_D=BLOCK_D,
            USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
        )
        grid = lambda args: (triton.cdiv(S_kv, args["BLOCK_N"]), B * H, 1)
        _fmha_bwd_kernel[grid](
            Q_desc,
            K_desc,
            V_desc,
            dO_desc,
            minus_L_desc,
            Delta_desc,
            dQ_desc,
            dq,
            dK_desc,
            dV_desc,
            ctx.sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            B,
            H,
            S_qo,
            S_kv,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            USE_DESC_FOR_CORR=USE_DESC_FOR_CORR,
        )
        return dq, dk, dv, None, None, None


triton_prefill_fmha = _attention.apply


@register_impl("fmha", backend="triton")
def triton_fmha(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    has_backward = kwargs.get("has_backward", False)
    o = triton_prefill_fmha(q, k, v, scaling, is_causal, has_backward)
    return o


# Backend Registration & Perf Markers

mark_perf_ready("fmha", "nvt")
