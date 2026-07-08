# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from tilegym.backend import get_available_triton_backend
from tilegym.backend import mark_perf_ready
from tilegym.backend import register_impl

# Constants & Type Aliases

INV_LOG_2 = tl.constexpr(1.0 / math.log(2))

# Autotune Config


def _get_mla_decoding_configs():
    """Get autotune configs based on GPU architecture and backend"""
    base_configs = [
        triton.Config(dict(BLOCK_H=64, BLOCK_N=64), num_ctas=1, num_stages=1),
        triton.Config(dict(BLOCK_H=64, BLOCK_N=128), num_ctas=1, num_stages=1),
    ]

    # Add Hopper+ specific config (SM90 and later)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        base_configs.append(triton.Config(dict(BLOCK_H=128, BLOCK_N=128), num_ctas=2, num_stages=1))

    return base_configs


# Device Kernels


@triton.autotune(
    configs=_get_mla_decoding_configs() if get_available_triton_backend() == "nvt" else [
        # OAIT meet CUDA error for BLOCK_H = 64, BLOCK_N = 64
        # use BLOCK_H = 32, BLOCK_N = 32 instead
        triton.Config(dict(BLOCK_H=32, BLOCK_N=32), num_ctas=1),
    ],
    key=["BLOCK_D", "BLOCK_KPE", "S_kv", "EVEN_N"],
)
@triton.heuristics({
    "EVEN_N": lambda args: args["S_kv"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _naive_absorb_mla(
    Q,
    QPE,
    KV,
    KPE,
    Out,
    L,
    sm_scale,
    stride_qb,
    stride_qm,
    stride_qpeb,
    stride_qpem,
    stride_kvb,
    stride_kvn,
    stride_kpeb,
    stride_kpem,
    stride_ob,
    stride_om,
    B,
    num_head,
    S_kv,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KPE: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    batch_idx = pid_y
    qk_scale = sm_scale * INV_LOG_2

    # Create tensor descriptors
    Q_desc = tl.make_tensor_descriptor(
        base=Q,
        shape=[B, num_head, BLOCK_D],
        strides=[stride_qb, stride_qm, 1],
        block_shape=[1, BLOCK_H, BLOCK_D],
    )
    QPE_desc = tl.make_tensor_descriptor(
        base=QPE,
        shape=[B, num_head, BLOCK_KPE],
        strides=[stride_qpeb, stride_qpem, 1],
        block_shape=[1, BLOCK_H, BLOCK_KPE],
    )
    K_desc = tl.make_tensor_descriptor(
        base=KV,
        shape=[B, S_kv, BLOCK_D],
        strides=[stride_kvb, stride_kvn, 1],
        block_shape=[1, BLOCK_N, BLOCK_D],
    )
    KPE_desc = tl.make_tensor_descriptor(
        base=KPE,
        shape=[B, S_kv, BLOCK_KPE],
        strides=[stride_kpeb, stride_kpem, 1],
        block_shape=[1, BLOCK_N, BLOCK_KPE],
    )
    V_desc = tl.make_tensor_descriptor(
        base=KV,
        shape=[B, S_kv, BLOCK_D],
        strides=[stride_kvb, stride_kvn, 1],
        block_shape=[1, BLOCK_N, BLOCK_D],
    )
    O_desc = tl.make_tensor_descriptor(
        base=Out,
        shape=[B, num_head, BLOCK_D],
        strides=[stride_ob, stride_om, 1],
        block_shape=[1, BLOCK_H, BLOCK_D],
    )
    L_desc = tl.make_tensor_descriptor(
        base=L,
        shape=[B, num_head],
        strides=[num_head, 1],
        block_shape=[1, BLOCK_H],
    )

    # Initialize accumulation variables
    m_prev = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_prev = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    # Load q
    q = Q_desc.load([batch_idx, pid_x * BLOCK_H, 0])
    q = tl.reshape(q, (BLOCK_H, BLOCK_D))
    qpe = QPE_desc.load([batch_idx, pid_x * BLOCK_H, 0])
    qpe = tl.reshape(qpe, (BLOCK_H, BLOCK_KPE))

    # Loop over key-value pairs and update accumulator
    end_n = S_kv
    cnt = 0
    offs_n = tl.arange(0, BLOCK_N)
    for curr_n in range(0, end_n, BLOCK_N):
        # Compute qk
        k = K_desc.load([batch_idx, cnt * BLOCK_N, 0])
        k = tl.reshape(k, (BLOCK_N, BLOCK_D))
        k = tl.trans(k, (1, 0))
        qk = tl.dot(q, k)
        kpe = KPE_desc.load([batch_idx, cnt * BLOCK_N, 0])
        kpe = tl.reshape(kpe, (BLOCK_N, BLOCK_KPE))
        kpe = tl.trans(kpe, (1, 0))
        qk = tl.dot(qpe, kpe, qk)
        # Apply mask to avoid out of bounds
        if not EVEN_N:
            mask = (curr_n + offs_n[None, :]) < S_kv
            qk = tl.where(mask, qk, -1.0e6)
        qk = qk.to(tl.float32)
        m_ij = tl.maximum(m_prev, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        # Attention weights
        p = tl.math.exp2(qk)
        l_curr = tl.sum(p, 1)
        alpha = tl.math.exp2(m_prev - m_ij)
        # Update m_prev and l_prev
        l_prev = l_prev * alpha + l_curr
        # Scale acc
        acc = acc * alpha[:, None]
        # Compute pv
        v = V_desc.load([batch_idx, cnt * BLOCK_N, 0])
        v = tl.reshape(v, (BLOCK_N, BLOCK_D))
        p = p.to(Q.dtype.element_ty)
        acc = tl.dot(p, v, acc)
        m_prev = m_ij
        cnt += 1

    acc = acc / (l_prev[:, None])
    l_prev = m_prev + tl.math.log2(l_prev)

    # Store results
    L_desc.store([batch_idx, pid_x * BLOCK_H], l_prev.reshape(1, BLOCK_H))

    acc = acc.to(Out.dtype.element_ty)
    O_desc.store([batch_idx, pid_x * BLOCK_H, 0], acc.reshape(1, BLOCK_H, BLOCK_D))


@triton.autotune(
    configs=[
        triton.Config(dict(BLOCK_H=16, BLOCK_N=128), num_ctas=1),
        triton.Config(dict(BLOCK_H=32, BLOCK_N=128), num_ctas=1),
    ] if get_available_triton_backend() == "nvt" else [
        # OAIT meet CUDA error when num_stages = 3
        # use num_stages = 1 instead
        triton.Config(dict(BLOCK_H=16, BLOCK_N=128), num_ctas=1, num_stages=1),
        triton.Config(dict(BLOCK_H=32, BLOCK_N=32), num_ctas=1),
    ],
    key=["BLOCK_D", "BLOCK_KPE", "S_kv", "EVEN_N"],
)
@triton.heuristics({
    "EVEN_N": lambda args: args["S_kv"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _naive_absorb_mla_transpose(
    Q,
    QPE,
    KV,
    KPE,
    Out,
    L,
    sm_scale,
    stride_qb,
    stride_qm,
    stride_qpeb,
    stride_qpem,
    stride_kvb,
    stride_kvn,
    stride_kpeb,
    stride_kpem,
    stride_ob,
    stride_om,
    B,
    num_head,
    S_kv,
    BLOCK_D: tl.constexpr,  # BLOCK_D = hidden_size
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KPE: tl.constexpr,  # BLOCK_KPE = position embedding size
    EVEN_N: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    batch_idx = pid_y
    qk_scale = sm_scale * INV_LOG_2

    # Create tensor descriptors
    Q_desc = tl.make_tensor_descriptor(
        base=Q,
        shape=[B, num_head, BLOCK_D],
        strides=[stride_qb, stride_qm, 1],
        block_shape=[1, BLOCK_H, BLOCK_D],
    )
    QPE_desc = tl.make_tensor_descriptor(
        base=QPE,
        shape=[B, num_head, BLOCK_KPE],
        strides=[stride_qpeb, stride_qpem, 1],
        block_shape=[1, BLOCK_H, BLOCK_KPE],
    )
    K_desc = tl.make_tensor_descriptor(
        base=KV,
        shape=[B, S_kv, BLOCK_D],
        strides=[stride_kvb, stride_kvn, 1],
        block_shape=[1, BLOCK_N, BLOCK_D],
    )
    KPE_desc = tl.make_tensor_descriptor(
        base=KPE,
        shape=[B, S_kv, BLOCK_KPE],
        strides=[stride_kpeb, stride_kpem, 1],
        block_shape=[1, BLOCK_N, BLOCK_KPE],
    )

    # TMA descriptor for V must be different from K,
    # to avoid its load being optimized out.
    V_desc = tl.make_tensor_descriptor(
        base=KV,
        shape=[S_kv, B, BLOCK_D],
        strides=[stride_kvn, stride_kvb, 1],
        block_shape=[BLOCK_N, 1, BLOCK_D],
    )
    O_desc = tl.make_tensor_descriptor(
        base=Out,
        shape=[B, num_head, BLOCK_D],
        strides=[stride_ob, stride_om, 1],
        block_shape=[1, BLOCK_H, BLOCK_D],
    )
    L_desc = tl.make_tensor_descriptor(
        base=L,
        shape=[B, num_head],
        strides=[num_head, 1],
        block_shape=[1, BLOCK_H],
    )

    # Initialize accumulation variables
    m_prev = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_prev = tl.full([BLOCK_N, BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_D, BLOCK_H], dtype=tl.float32)

    # Load query and query position encoding
    q = Q_desc.load([batch_idx, pid_x * BLOCK_H, 0])
    q = tl.reshape(q, (BLOCK_H, BLOCK_D))
    q = tl.trans(q, (1, 0))
    qpe = QPE_desc.load([batch_idx, pid_x * BLOCK_H, 0])
    qpe = tl.reshape(qpe, (BLOCK_H, BLOCK_KPE))
    qpe = tl.trans(qpe, (1, 0))

    # Loop over key-value pairs and update accumulator
    end_n = S_kv
    cnt = 0
    mask_start = S_kv // BLOCK_N * BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    for curr_n in range(0, end_n, BLOCK_N):
        # Load key and compute Q@K^T
        k = K_desc.load([batch_idx, cnt * BLOCK_N, 0])
        k = tl.reshape(k, (BLOCK_N, BLOCK_D))
        qk = tl.dot(k, q)

        # Load key position encoding and compute QPE@KPE^T
        kpe = KPE_desc.load([batch_idx, cnt * BLOCK_N, 0])
        kpe = tl.reshape(kpe, (BLOCK_N, BLOCK_KPE))
        qk = tl.dot(kpe, qpe, qk)

        if curr_n >= mask_start:
            mask = (curr_n + offs_n) < S_kv
            qk = tl.where(mask[:, None], qk, -1.0e6)

        # Apply scaling and compute attention scores
        qk = qk.to(tl.float32)
        m_ij = tl.maximum(m_prev, tl.max(qk, 0) * qk_scale)
        qk = qk * qk_scale - m_ij[None, :]

        # Compute attention weights and update running statistics
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_prev - m_ij)
        l_prev = l_prev * alpha[None, :] + p
        acc = acc * alpha[None, :]

        # Load value and compute attention @ value
        v = V_desc.load([cnt * BLOCK_N, batch_idx, 0])
        v = tl.reshape(v, (BLOCK_N, BLOCK_D))
        v = tl.trans(v, (1, 0))
        p = p.to(Q.dtype.element_ty)
        acc = tl.dot(v, p, acc)
        m_prev = m_ij
        cnt += 1

    # Finalize attention computation
    l_prev = tl.sum(l_prev, 0)
    acc = acc / (l_prev[None, :])
    l_prev = m_prev + tl.math.log2(l_prev)

    # Store results
    L_desc.store([batch_idx, pid_x * BLOCK_H], l_prev.reshape(1, BLOCK_H))

    acc = acc.to(Out.dtype.element_ty)
    acc = tl.trans(acc, (1, 0))
    O_desc.store([batch_idx, pid_x * BLOCK_H, 0], acc.reshape(1, BLOCK_H, BLOCK_D))


# Host Launchers & Public API


class _mla_decoding(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        qpe,
        kv,
        kpe,
        sm_scale,
        transpose,
    ):
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        # Setup stride and shape
        B, num_head, BLOCK_D = q.shape
        BLOCK_KPE = kpe.shape[2]
        S_kv = kv.shape[1]
        o = torch.empty_like(q)
        l = torch.empty((B, num_head), device=q.device, dtype=torch.float32).contiguous()

        # Launch fmha fwd kernel
        grid = lambda META: (triton.cdiv(num_head, META["BLOCK_H"]), B, 1)
        if not transpose:
            kernel = _naive_absorb_mla
        else:
            kernel = _naive_absorb_mla_transpose

        kernel[grid](
            q,
            qpe,
            kv,
            kpe,
            o,
            l,
            sm_scale,
            q.stride(0),
            q.stride(1),
            qpe.stride(0),
            qpe.stride(1),
            kv.stride(0),
            kv.stride(1),
            kpe.stride(0),
            kpe.stride(1),
            o.stride(0),
            o.stride(1),
            B,
            num_head,
            S_kv,
            BLOCK_D=BLOCK_D,
            BLOCK_KPE=BLOCK_KPE,
        )
        return o, l

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError()


class MLADecoding:

    def __init__(self, transpose):
        self.transpose = transpose

    def __call__(self, q, qpe, kv, kpe, sm_scale):
        if sm_scale is None:
            sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))
        o, l = _mla_decoding.apply(
            q,
            qpe,
            kv,
            kpe,
            sm_scale,
            self.transpose,
        )
        return o, l


@register_impl("mla_decoding", backend="triton")
def mla_decoding(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: float,
    transpose: bool = True,
    **kwargs,
) -> torch.Tensor:
    return _mla_decoding.apply(q, qpe, kv, kpe, sm_scale, transpose)


# Backend Registration & Perf Markers

mark_perf_ready("mla_decoding", "nvt")
