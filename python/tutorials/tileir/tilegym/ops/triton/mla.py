# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
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

# Capability Probe


def _supports_host_descriptor():
    return torch.cuda.get_device_capability()[0] >= 9


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
    KPE_desc,
    V_desc,
    acc,
    l_i,
    m_i,
    q,
    qpe,
    batch_idx,
    off_kv_h,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KPE: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    EVEN_K: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
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
    for curr_n in range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        curr_n = tl.multiple_of(curr_n, BLOCK_N)
        # Compute qk
        # Calculate K offsets - we load BLOCK_N x BLOCK_D and then transpose
        k_offset_b = batch_idx * 1
        k_offset_h = off_kv_h * 1
        k_offset_n = cnt * BLOCK_N
        k_offset_d = 0 * BLOCK_D
        k = K_desc.load([k_offset_b, k_offset_h, k_offset_n, k_offset_d])
        k = tl.reshape(k, (BLOCK_N, BLOCK_D))
        # Transpose K to get (BLOCK_D, BLOCK_N) for matrix multiplication
        k = tl.trans(k)
        qk = tl.dot(q, k)

        # Calculate KPE offsets
        kpe_offset_b = batch_idx * 1
        kpe_offset_h = 0 * 1
        kpe_offset_n = cnt * BLOCK_N
        kpe_offset_d = 0 * BLOCK_KPE
        # Add position embedding contribution
        kpe = KPE_desc.load([kpe_offset_b, kpe_offset_h, kpe_offset_n, kpe_offset_d])
        kpe = tl.reshape(kpe, (BLOCK_N, BLOCK_KPE))
        # Transpose KPE to get (BLOCK_KPE, BLOCK_N) for matrix multiplication
        kpe = tl.trans(kpe)
        qk = tl.dot(qpe, kpe, qk)

        # Process boundary case(here we only need to process non-causal case)
        # Apply mask
        if STAGE == 3 and not EVEN_K:
            mask = curr_n + offs_n[None, :] < N_CTX
            qk = tl.where(mask, qk, -1.0e6)
        if STAGE == 2:
            mask = offs_m[:, None] >= (curr_n + offs_n[None, :])
            qk = tl.where(mask, qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)

        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        # Scale acc
        acc = acc * alpha[:, None]

        # Compute pv
        v_offset_b = batch_idx * 1
        v_offset_h = off_kv_h * 1
        v_offset_n = cnt * BLOCK_N
        v_offset_d = 0 * BLOCK_D
        v = V_desc.load([v_offset_b, v_offset_h, v_offset_n, v_offset_d])
        v = tl.reshape(v, (BLOCK_N, BLOCK_D))
        p = p.to(q.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        cnt += 1

    return acc, l_i, m_i


# Autotune Config


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    BLOCK_KPE = nargs["BLOCK_KPE"]
    if not isinstance(nargs["Q"], TensorDescriptor):
        return
    nargs["Q"].block_shape = [1, 1, BLOCK_M, BLOCK_D]
    nargs["QPE"].block_shape = [1, 1, BLOCK_M, BLOCK_KPE]
    nargs["K"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
    nargs["KPE"].block_shape = [1, 1, BLOCK_N, BLOCK_KPE]
    nargs["V"].block_shape = [1, 1, BLOCK_N, BLOCK_D]
    nargs["Out"].block_shape = [1, 1, BLOCK_M, BLOCK_D]


def _get_configs(kernel_type="prefill"):
    if _supports_host_descriptor():
        _hook = _host_descriptor_pre_hook
    else:
        _hook = None

    def keep(conf):
        """Filter out invalid configurations for the given kernel type.

        For prefill kernels with causal attention, BLOCK_M must be >= BLOCK_N.
        This is because the diagonal block loop (STAGE 2) iterates with step BLOCK_N
        over a range of size BLOCK_M. If BLOCK_M < BLOCK_N, the loop executes
        0 iterations, causing incorrect results for all blocks after the first.

        The constraint BLOCK_M % BLOCK_N == 0 ensures BLOCK_M is a multiple of
        BLOCK_N (and thus BLOCK_M >= BLOCK_N).
        """
        BLOCK_M = conf.kwargs["BLOCK_M"]
        BLOCK_N = conf.kwargs["BLOCK_N"]
        if kernel_type == "prefill":
            return BLOCK_M % BLOCK_N == 0
        else:
            return True

    if get_available_triton_backend() == "nvt":
        default_config = {"warp_specialize": False}
        if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
            return [triton.Config(dict(BLOCK_M=64, BLOCK_N=64, occupancy=2, **default_config), pre_hook=_hook)]
        elif torch.cuda.get_device_capability() == (9, 0):
            sm90_configs = [
                # occupancy = 2. We currently do not expose num_warps in CudaTile.
                # Once we do (and if we do), we can experiment with occupancy = 1 and num_warps = 8.
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, **default_config), pre_hook=_hook)
                for BM in [64, 128]
                for BN in [64, 128]
            ]
            return list(filter(keep, sm90_configs))
        elif torch.cuda.get_device_capability()[0] == 8:
            # SM80 (A100): smaller tiles reduce register pressure and improve occupancy.
            sm80_configs = [
                triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=occ, **default_config), pre_hook=_hook)
                for BM in [64, 128]
                for BN in [64, 128]
                for occ in [1, 2]
            ]
            return list(filter(keep, sm80_configs))
        else:
            # SM100 (Blackwell): expand search over BLOCK sizes, occupancy, num_ctas, num_stages.
            # Dot-related kernel → 2CTA available; per-Config pre_hook required for TMA.
            sm100_configs = []
            # 1CTA configs across block sizes and occupancy
            for BM, BN in [(128, 128), (256, 128), (128, 64), (64, 64)]:
                for occ in [1, 2]:
                    for ns in [2, 3, 4]:
                        sm100_configs.append(
                            triton.Config(
                                dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=occ, **default_config),
                                num_stages=ns,
                                pre_hook=_hook,
                            ))
            # 2CTA configs (SM100+ only) - dot-related kernel benefits from 2CTA MMA
            for BM, BN in [(256, 128), (128, 128), (256, 64), (256, 256), (128, 64)]:
                for ns in [3, 4, 5, 6]:
                    sm100_configs.append(
                        triton.Config(
                            dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, **default_config),
                            num_stages=ns,
                            num_ctas=2,
                            pre_hook=_hook,
                        ))
            # 2CTA configs with warp_specialize=True around the winning shape
            for BM, BN in [(256, 128), (128, 128), (256, 256)]:
                for ns in [4, 5, 6]:
                    sm100_configs.append(
                        triton.Config(
                            dict(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, warp_specialize=True),
                            num_stages=ns,
                            num_ctas=2,
                            pre_hook=_hook,
                        ))
            return list(filter(keep, sm100_configs))

    # Full tuning space for oait
    configs = [
        triton.Config(dict(BLOCK_M=BM, BLOCK_N=BN, warp_specialize=ws), num_stages=s, num_warps=w, pre_hook=_hook)
        for BM in [64, 128, 256]
        for BN in [64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
        for ws in [True, False]
    ]
    configs = list(filter(keep, configs))
    return configs


# Device Kernels


@triton.autotune(
    configs=_get_configs(),
    key=["S_qo", "S_kv", "STAGE", "QUERY_GROUP_SIZE", "dtype"],
)
@triton.heuristics({
    "EVEN_K": lambda args: args["S_kv"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _prefill_mla(
    Q,
    QPE,
    K,
    KPE,
    V,
    Out,
    L,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qpeb,
    stride_qpeh,
    stride_qpem,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kpeb,
    stride_kpeh,
    stride_kpen,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    B,
    H,
    S_qo,
    S_kv,
    BLOCK_D: tl.constexpr,  # BLOCK_D = hidden_size
    BLOCK_KPE: tl.constexpr,  # BLOCK_KPE = position embedding size
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

    QPE_desc = _maybe_make_tensor_desc(
        QPE,
        shape=[B, H, S_qo, BLOCK_KPE],
        strides=[stride_qpeb, stride_qpeh, stride_qpem, 1],
        block_shape=[1, 1, BLOCK_M, BLOCK_KPE],
    )

    # For KPE, we need to load BLOCK_N x BLOCK_KPE but then transpose to BLOCK_KPE x BLOCK_N
    KPE_desc = _maybe_make_tensor_desc(
        KPE,
        shape=[B, 1, S_qo, BLOCK_KPE],
        strides=[stride_kpeb, stride_kpeh, stride_kpen, 1],
        block_shape=[1, 1, BLOCK_N, BLOCK_KPE],
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

    L_desc = _maybe_make_tensor_desc(
        L,
        shape=[B, H, S_qo],
        strides=[1, 1, 1],
        block_shape=[1, 1, BLOCK_M],
    )

    # Initialize m, l, acc
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q
    q_offset_b = batch_idx * 1
    q_offset_h = head_idx * 1
    q_offset_m = pid_x * BLOCK_M
    q_offset_d = 0 * BLOCK_D
    q = Q_desc.load([q_offset_b, q_offset_h, q_offset_m, q_offset_d])
    q = tl.reshape(q, (BLOCK_M, BLOCK_D))

    # Load qpe
    qpe_offset_b = batch_idx * 1
    qpe_offset_h = head_idx * 1
    qpe_offset_m = pid_x * BLOCK_M
    qpe_offset_d = 0 * BLOCK_KPE
    qpe = QPE_desc.load([qpe_offset_b, qpe_offset_h, qpe_offset_m, qpe_offset_d])
    qpe = tl.reshape(qpe, (BLOCK_M, BLOCK_KPE))

    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            K_desc,
            KPE_desc,
            V_desc,
            acc,
            l_i,
            m_i,
            q,  #
            qpe,
            batch_idx,
            off_kv_h,
            pid_x,
            qk_scale,  #
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,  #
            BLOCK_KPE,
            4 - STAGE,
            offs_m,
            offs_n,
            EVEN_K,
            S_kv,
            warp_specialize,
        )
    # Stage 2: on-band
    if STAGE & 2:
        # Barrier makes it easier for compielr to schedule the
        # Two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            K_desc,
            KPE_desc,
            V_desc,
            acc,
            l_i,
            m_i,
            q,
            qpe,
            batch_idx,
            off_kv_h,
            pid_x,
            qk_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            BLOCK_KPE,
            2,
            offs_m,
            offs_n,
            EVEN_K,
            S_kv,
            warp_specialize,
        )
    # Epilogue
    acc = acc / (l_i[:, None])
    l_i = m_i + tl.math.log2(l_i)

    # Write back l and o - calculate explicit offsets
    l_offset_b = batch_idx * 1
    l_offset_h = head_idx * 1
    l_offset_m = pid_x * BLOCK_M
    L_desc.store([l_offset_b, l_offset_h, l_offset_m], l_i.reshape(1, 1, BLOCK_M))

    o_offset_b = batch_idx * 1
    o_offset_h = head_idx * 1
    o_offset_m = pid_x * BLOCK_M
    o_offset_d = 0 * BLOCK_D
    O_desc.store(
        [o_offset_b, o_offset_h, o_offset_m, o_offset_d],
        acc.to(dtype).reshape(1, 1, BLOCK_M, BLOCK_D),
    )


# Host Launchers & Public API


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, qpe, k, kpe, v, sm_scale, IS_CAUSAL):
        # Setup stride and shape
        B, H, S_qo, BLOCK_D = q.shape
        BLOCK_KPE = kpe.shape[3]
        assert k.shape == v.shape
        num_head_kv = k.shape[1]
        S_kv = k.shape[2]
        o = torch.empty_like(q)
        l = torch.empty((B, H, S_qo), device=q.device, dtype=torch.float32)

        if H == num_head_kv:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)
        # Launch fmha fwd kernel
        grid = lambda args: (triton.cdiv(S_qo, args["BLOCK_M"]), B * H, 1)

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)
        if _supports_host_descriptor():
            dummy_block_qo = [1, 1, 1, BLOCK_D]
            dummy_block_kv = [1, 1, 1, BLOCK_D]
            dummy_block_qpe = [1, 1, 1, BLOCK_KPE]
            dummy_block_kpe = [1, 1, 1, BLOCK_KPE]
            desc_q = TensorDescriptor(
                q,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[q.stride(0), q.stride(1), q.stride(2), 1],
                block_shape=dummy_block_qo,
            )
            desc_qpe = TensorDescriptor(
                qpe,
                shape=[B, H, S_qo, BLOCK_KPE],
                strides=[qpe.stride(0), qpe.stride(1), qpe.stride(2), 1],
                block_shape=dummy_block_qpe,
            )
            desc_k = TensorDescriptor(
                k,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[k.stride(0), k.stride(1), k.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            desc_kpe = TensorDescriptor(
                kpe,
                shape=[B, 1, S_qo, BLOCK_KPE],
                strides=[kpe.stride(0), kpe.stride(1), kpe.stride(2), 1],
                block_shape=dummy_block_kpe,
            )
            desc_v = TensorDescriptor(
                v,
                shape=[B, H, S_kv, BLOCK_D],
                strides=[v.stride(0), v.stride(1), v.stride(2), 1],
                block_shape=dummy_block_kv,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[B, H, S_qo, BLOCK_D],
                strides=[o.stride(0), o.stride(1), o.stride(2), 1],
                block_shape=dummy_block_qo,
            )
        else:
            desc_q = q
            desc_qpe = qpe
            desc_k = k
            desc_kpe = kpe
            desc_v = v
            desc_o = o

        def torch_to_triton_type(dtype):
            type_map = {
                torch.float16: tl.float16,
                torch.float8_e5m2: tl.float8e5,
                torch.float32: tl.float32,
                torch.int32: tl.int32,
                torch.bfloat16: tl.bfloat16,
            }
            return type_map.get(dtype, tl.float16)

        dtype = torch_to_triton_type(q.dtype)
        # disable autotune when run test_op in test_mla.py
        if is_autotune_disabled():
            _prefill_mla.configs = [_prefill_mla.configs[0]]
        _prefill_mla[grid](
            desc_q,
            desc_qpe,
            desc_k,
            desc_kpe,
            desc_v,
            desc_o,
            l,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            qpe.stride(0),
            qpe.stride(1),
            qpe.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kpe.stride(0),
            kpe.stride(1),
            kpe.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            B,
            H,
            S_qo,
            S_kv,
            BLOCK_D=BLOCK_D,
            BLOCK_KPE=BLOCK_KPE,
            STAGE=3 if IS_CAUSAL else 1,
            QUERY_GROUP_SIZE=query_group_size,
            dtype=dtype,
        )
        ctx.save_for_backward(q, k, v, o, l)
        ctx.sm_scale = sm_scale
        ctx.shapes = (B, H, S_qo, S_kv)
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError()


class _Attention:

    def __init__(self, IS_CAUSAL):
        self.IS_CAUSAL = IS_CAUSAL

    def __call__(self, q, k, v, sm_scale, qpe=None, kpe=None):
        c = _attention.apply(
            q,
            qpe,
            k,
            kpe,
            v,
            sm_scale,
            self.IS_CAUSAL,
        )
        return c


@register_impl("mla", backend="triton")
def triton_mla(q, k, v, qpe, kpe, is_causal, scaling, **kwargs):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

    attention = _Attention(is_causal)
    o = attention(q, k, v, scaling, qpe, kpe)
    return o


# Backend Registration & Perf Markers

mark_perf_ready("mla", "nvt")
