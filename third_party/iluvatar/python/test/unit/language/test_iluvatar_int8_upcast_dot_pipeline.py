"""Regression test for the ``int8 -> upcast -> tl.dot`` path on Iluvatar GPU
when the dot is inside a pipelined loop. The bug originally surfaced as a
segfault only when ``num_stages > 1`` and the upcast targeted ``fp32``; the
``i8 -> bf16/fp16`` variants are kept as positive controls.
"""

import math

import pytest
import torch

import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA / Iluvatar GPU")

LOG2E = tl.constexpr(math.log2(math.e))


@triton.jit
def _flash_attn_int8kv_kernel(
    Q,
    K_cache,
    V_cache,
    Out,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ot,
    stride_od,
    seq_len,
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CAST_DTYPE: tl.constexpr,
):
    Q_blk = tl.make_block_ptr(
        base=Q,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(stride_qt, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(Q_blk, boundary_check=(0, 1), padding_option="zero").to(CAST_DTYPE)

    m_i = tl.zeros((BLOCK_M, ), dtype=tl.float32) - float("inf")
    l_i = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    for page_idx in tl.range(0, num_pages):
        K_T_blk = tl.make_block_ptr(
            base=K_cache + page_idx * stride_kb,
            shape=(HEAD_DIM, PAGE_SIZE),
            strides=(stride_kd, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D, BLOCK_N),
            order=(0, 1),
        )
        V_blk = tl.make_block_ptr(
            base=V_cache + page_idx * stride_vb,
            shape=(PAGE_SIZE, HEAD_DIM),
            strides=(stride_vn, stride_vd),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        k_T = tl.load(K_T_blk, boundary_check=(0, 1), padding_option="zero").to(CAST_DTYPE)
        v = tl.load(V_blk, boundary_check=(0, 1), padding_option="zero").to(CAST_DTYPE)

        qk = tl.dot(q, k_T) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(LOG2E * (m_i - m_ij))
        p = tl.math.exp2(LOG2E * (qk - m_ij[:, None]))

        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(CAST_DTYPE), v, acc=acc)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_ij

    l_i = tl.maximum(l_i, 1e-6)
    acc = acc / l_i[:, None]

    O_blk = tl.make_block_ptr(
        base=Out,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(stride_ot, stride_od),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(O_blk, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))


def _flash_attn_int8kv_ref(q_bf16, k_int8, v_int8, sm_scale, cast_dtype):
    q = q_bf16.to(cast_dtype).float()
    k = k_int8.reshape(-1, k_int8.shape[-1]).to(cast_dtype).float()
    v = v_int8.reshape(-1, v_int8.shape[-1]).to(cast_dtype).float()
    qk = (q @ k.T) * sm_scale
    p = torch.softmax(qk, dim=-1)
    return (p @ v).to(torch.bfloat16)


_TORCH_TO_TL = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}


@pytest.mark.parametrize("cast_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_stages", [1, 2, 3])
@pytest.mark.parametrize("seq_len", [64, 128, 256])
def test_flash_attn_int8_upcast_dot_pipeline(num_stages, seq_len, cast_dtype):
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 128
    HEAD_DIM, PAGE_SIZE = 128, 64
    assert seq_len % PAGE_SIZE == 0

    torch.manual_seed(0)
    q = torch.randn(BLOCK_M, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randint(
        -128,
        127,
        (seq_len // PAGE_SIZE, PAGE_SIZE, HEAD_DIM),
        dtype=torch.int8,
        device="cuda",
    )
    v_cache = torch.randint(
        -128,
        127,
        (seq_len // PAGE_SIZE, PAGE_SIZE, HEAD_DIM),
        dtype=torch.int8,
        device="cuda",
    )
    out = torch.empty(BLOCK_M, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    _flash_attn_int8kv_kernel[(1, )](
        q,
        k_cache,
        v_cache,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        out.stride(0),
        out.stride(1),
        seq_len,
        float(sm_scale),
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        PAGE_SIZE=PAGE_SIZE,
        CAST_DTYPE=_TORCH_TO_TL[cast_dtype],
        num_stages=num_stages,
    )
    torch.cuda.synchronize()

    ref = _flash_attn_int8kv_ref(q, k_cache, v_cache, sm_scale, cast_dtype)
    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    max_abs = diff.max().item()
    mean_rel = rel.mean().item()
    assert max_abs <= 0.5 and mean_rel < 0.1, (f"flash-attn int8kv mismatch with num_stages={num_stages}, "
                                               f"seq_len={seq_len}, cast_dtype={cast_dtype}: "
                                               f"max abs diff={max_abs}, mean rel diff={mean_rel}")
