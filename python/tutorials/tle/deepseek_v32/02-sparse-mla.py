# flagtree
"""
Sparse MLA Forward
==================

This tutorial provides:
- Triton sparse MLA forward kernel (no TLE API in kernel body)
- Triton+TLE sparse MLA forward kernel (shared-memory staging)
- optional TileLang sparse MLA forward kernel (inlined from TileLang example)
- correctness test and benchmark entry
"""

import argparse
import math

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

try:
    import tilelang
    from tilelang import language as T

    _HAVE_TILELANG = True
except Exception:  # pragma: no cover - optional dependency
    tilelang = None
    T = None
    _HAVE_TILELANG = False

spar_mla_fwd_configs = [
    triton.Config({"num_stages": 1, "num_warps": 4}),
    triton.Config({"num_stages": 1, "num_warps": 8}),
    triton.Config({"num_stages": 1, "num_warps": 16}),
    triton.Config({"num_stages": 1, "num_warps": 32}),
    triton.Config({"num_stages": 2, "num_warps": 4}),
    triton.Config({"num_stages": 2, "num_warps": 8}),
    triton.Config({"num_stages": 2, "num_warps": 16}),
    triton.Config({"num_stages": 2, "num_warps": 32}),
    triton.Config({"num_stages": 4, "num_warps": 4}),
    triton.Config({"num_stages": 4, "num_warps": 8}),
    triton.Config({"num_stages": 4, "num_warps": 16}),
    triton.Config({"num_stages": 4, "num_warps": 32}),
]
tle_spar_mla_fwd_configs = [
    triton.Config({"num_stages": 2, "num_warps": 4}),
    triton.Config({"num_stages": 2, "num_warps": 8}),
    triton.Config({"num_stages": 2, "num_warps": 16}),
    triton.Config({"num_stages": 2, "num_warps": 32}),
]


@triton.autotune(
    configs=spar_mla_fwd_configs,
    key=["SQ", "K", "H", "D", "is_causal"],
)
@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kvb,
    stride_kvg,
    stride_kvn,
    stride_kvd,
    stride_tb,
    stride_tg,
    stride_tm,
    stride_tt,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    B: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    VG: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_h[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, other=0.0)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0)

    max_prev = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float32)
    acc = tl.zeros([BH, DP], dtype=tl.float32)

    log_scale: tl.constexpr = sm_scale * 1.44269504
    max_col = i_sq if is_causal else SQ - 1

    NK = tl.cdiv(K, BK)
    for ck in tl.range(NK, num_stages=0):
        if ck * BK <= max_col:
            t_ptr = (BK * ck + offs_t) * stride_tt
            t_msk = t_ptr < K
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)

            kv_ptr = kv_base + offs_d[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            kv_msk = mask_d[:, None] & mask_ids[None, :]
            kv_blk = tl.load(kv_ptr, kv_msk, other=0.0)

            tkv_ptr = tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            tkv_msk = mask_td[:, None] & mask_ids[None, :]
            tkv_blk = tl.load(tkv_ptr, tkv_msk, other=0.0)

            qk = tl.dot(tq_blk, tkv_blk, out_dtype=tl.float32)
            qk = tl.dot(q_blk, kv_blk, qk, out_dtype=tl.float32)
            qk = tl.where(mask_ids[None, :], qk, float("-inf"))

            new_max = tl.maximum(max_prev, tl.max(qk, axis=1))
            alpha = tl.math.exp2((max_prev - new_max) * log_scale)
            exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)
            sum_qk = tl.sum(exp_qk, axis=1)
            sum_exp = sum_exp * alpha + sum_qk
            acc = acc * alpha[:, None]
            exp_qk = exp_qk.to(tl.bfloat16)
            acc = tl.dot(exp_qk, tl.trans(kv_blk), acc, out_dtype=tl.float32)
            max_prev = new_max

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_prev * log_scale + tl.math.log2(sum_exp.to(tl.float32))
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)


@triton.autotune(
    configs=tle_spar_mla_fwd_configs,
    key=["SQ", "K", "H", "D", "is_causal"],
)
@triton.jit
def tle_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kvb,
    stride_kvg,
    stride_kvn,
    stride_kvd,
    stride_tb,
    stride_tg,
    stride_tm,
    stride_tt,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    B: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    VG: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr,
):
    # TileLang-style forward path:
    # - stage Q and Q_tail once in shared memory;
    # - load sparse KV/K_tail blocks directly from global memory per K tile;
    # - online softmax on logits;
    # - use probabilities directly for the second GEMM.
    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    q_base = q + i_b * stride_qb + i_sq * stride_qm + i_gbh * (BH * stride_qh)
    tq_base = q_base + D * stride_qd
    kv_base = kv + i_b * stride_kvb + i_g * stride_kvg
    tkv_base = kv_base + D * stride_kvd
    t_base = indices + i_b * stride_tb + i_sq * stride_tm + i_g * stride_tg
    o_base = output + i_b * stride_ob + i_sq * stride_om + i_gbh * (BH * stride_oh)
    l_base = lse + i_b * stride_lb + i_sq * stride_lm + i_gbh * (BH * stride_lh)

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_msk = mask_h[:, None] & mask_d[None, :]
    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :] * stride_qd
    tq_msk = mask_h[:, None] & mask_td[None, :]

    q_smem = tle.gpu.alloc(
        [BH, DP],
        dtype=q.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    tq_smem = tle.gpu.alloc(
        [BH, TDP],
        dtype=q.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )

    q_smem_ptr = tle.gpu.local_ptr(q_smem)
    q_blk = tl.load(q_ptr, q_msk, other=0.0)
    tl.store(q_smem_ptr, q_blk)

    tq_smem_ptr = tle.gpu.local_ptr(tq_smem)
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0)
    tl.store(tq_smem_ptr, tq_blk)

    max_prev = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float32)
    acc = tl.zeros([BH, DP], dtype=tl.float32)

    log_scale: tl.constexpr = sm_scale * 1.44269504
    max_col = i_sq if is_causal else SQ - 1

    NK = tl.cdiv(K, BK)
    for ck in tl.range(NK, num_stages=2):
        if ck * BK <= max_col:
            t_ptr = (BK * ck + offs_t) * stride_tt
            t_msk = t_ptr < K
            t_ptr += t_base
            kv_ids = tl.load(t_ptr, t_msk, other=-1)
            mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)
            kv_ids_safe = tl.where(mask_ids, kv_ids, 0)

            kv_ptr = kv_base + offs_d[:, None] * stride_kvd + kv_ids_safe[None, :] * stride_kvn
            kv_col = tl.load(kv_ptr, mask=mask_d[:, None], other=0.0)

            tkv_ptr = tkv_base + offs_td[:, None] * stride_kvd + kv_ids_safe[None, :] * stride_kvn
            tkv_col = tl.load(tkv_ptr, mask=mask_td[:, None], other=0.0)

            tq_blk = tl.load(tq_smem_ptr)
            qk = tl.dot(tq_blk, tkv_col, out_dtype=tl.float32)
            q_blk = tl.load(q_smem_ptr)
            qk = tl.dot(q_blk, kv_col, qk, out_dtype=tl.float32)
            qk = tl.where(mask_ids[None, :], qk, float("-inf"))

            new_max = tl.maximum(max_prev, tl.max(qk, axis=1))
            alpha = tl.math.exp2((max_prev - new_max) * log_scale)
            exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)
            sum_qk = tl.sum(exp_qk, axis=1)
            sum_exp = sum_exp * alpha + sum_qk
            acc = acc * alpha[:, None]
            exp_qk = exp_qk.to(q.dtype.element_ty)
            acc = tl.dot(exp_qk, tl.trans(kv_col), acc, out_dtype=tl.float32)
            max_prev = new_max

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q.dtype.element_ty), o_msk)

    fin_log = max_prev * log_scale + tl.math.log2(sum_exp.to(tl.float32))
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q.dtype.element_ty), l_msk)


def _sparse_mla_fwd_interface_impl(kernel, q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512, bk=32):
    is_causal = True
    assert not return_p_sum, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, S, VG, _ = kv.shape

    D = d_v
    assert kv.shape[-1] == DT
    TD = DT - D
    DP = triton.next_power_of_2(D)
    TDP = triton.next_power_of_2(TD)
    assert kv.shape[0] == B
    _, _, _, K = indices.shape
    assert indices.shape == (B, SQ, VG, K)
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    BH = 32
    NH = triton.cdiv(G, BH)
    BK = bk
    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)
    grid = (B, SQ, VG * NH)
    kernel[grid](
        q,
        kv,
        indices,
        sm_scale,
        output,
        lse,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),
        kv.stride(0),
        kv.stride(2),
        kv.stride(1),
        kv.stride(3),
        indices.stride(0),
        indices.stride(2),
        indices.stride(1),
        indices.stride(3),
        output.stride(0),
        output.stride(2),
        output.stride(1),
        output.stride(3),
        lse.stride(0),
        lse.stride(2),
        lse.stride(1),
        B,
        SQ,
        S,
        K,
        D,
        TD,
        DP,
        TDP,
        H,
        G,
        VG,
        BK,
        BH,
        is_causal,
    )
    return output, lse


def triton_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    return _sparse_mla_fwd_interface_impl(
        triton_sparse_mla_fwd,
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        return_p_sum=return_p_sum,
        d_v=d_v,
        bk=32,
    )


def tle_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    return _sparse_mla_fwd_interface_impl(
        tle_sparse_mla_fwd,
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        return_p_sum=return_p_sum,
        d_v=d_v,
        bk=64,
    )


if _HAVE_TILELANG:

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def tilelang_sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        block_I=64,
        num_stages=2,
        threads=256,
    ):
        assert dim == tilelang.math.next_power_of_2(dim), f"dim should be power-of-2, but got {dim}"
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim should be power-of-2, but got {tail_dim}"
        assert is_causal, "non-causal path is not implemented"
        assert topk % block_I == 0, "topk should be divisible by block_I"
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504
        else:
            sm_scale = sm_scale * 1.44269504

        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = T.int32
        dtype = T.bfloat16
        accum_dtype = T.float32

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1, "automatic head padding only supports kv_group == 1"

        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        D_tail = tail_dim

        if head_kv > 64:
            assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
            replicate_h = head_kv // 64
        else:
            replicate_h = 1

        H_per_block = padded_H if replicate_h == 1 else 64

        @T.prim_func
        def main(Q: T.Tensor(q_shape, dtype),  # type: ignore
                 KV: T.Tensor(kv_shape, dtype),  # type: ignore
                 Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
                 Output: T.Tensor(o_shape, dtype),  # type: ignore
                 Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
                 ):
            with T.Kernel(seq_len * replicate_h, batch, kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
            ):
                Q_shared = T.alloc_shared([H_per_block, D], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")
                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))

                b_i, g_i = by, bz
                s_i = bx if replicate_h == 1 else (bx // replicate_h)
                q_i = s_i
                max_kv_i = q_i

                H0 = g_i * padded_H + (0 if replicate_h == 1 else (bx % replicate_h) * 64)
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                for i_i in T.Pipelined(NI, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                    for bi_i, d_i in T.Parallel(BI, D):
                        KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
                T.copy(sumexp, Lse[b_i, s_i, H0:H1])

        return main

else:
    tilelang_sparse_mla_fwd = None


def tilelang_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256,
):
    if not _HAVE_TILELANG or tilelang_sparse_mla_fwd is None:
        raise RuntimeError("TileLang is not installed, cannot run TileLang sparse MLA bench")

    is_causal = True
    assert not return_p_sum, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, _seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert dim == triton.next_power_of_2(dim), f"d_v should be power-of-2 for TileLang path, but got {dim}"
    assert tail_dim == triton.next_power_of_2(
        tail_dim), f"tail dim should be power-of-2 for TileLang path, but got {tail_dim}"
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = tilelang_sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_causal,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, indices)
    return out, lse


def _build_sparse_mla_inputs(B=1, S=4096, SKV=4096, H=128, HKV=1, DQK=576, topk=2048, dtype=torch.bfloat16, seed=0):
    torch.random.manual_seed(seed)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i
    return q, kv, indices


def _resolve_tilelang_block_i(topk: int, block_i: int) -> int:
    if block_i <= 0:
        raise ValueError(f"tilelang block_I should be > 0, but got {block_i}")
    if topk % block_i == 0:
        return block_i
    fallback = math.gcd(topk, block_i)
    if fallback <= 0:
        raise ValueError(f"cannot find a valid tilelang block_I for topk={topk}, block_I={block_i}")
    return fallback


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=512):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = d_v
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32,
                                          device="cuda").view(-1,
                                                              1) >= torch.arange(1 - 1, sk * 1, 1, dtype=torch.int32,
                                                                                 device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.bfloat16)


def _sparse_mla_tflops(B, S, H, DQK, DV, topk, ms):
    return (B * S * (DQK + DV) * topk * 2 * H) / (ms * 1e-3) / 1e12


def _bench_ms(fn, warmup=200, rep=100):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return float(ms if not isinstance(ms, tuple) else ms[0])


_BENCH_PROVIDERS = (["triton"] + ["tle"] + (["tilelang"] if _HAVE_TILELANG else []))
_BENCH_NAMES = (["Triton"] + ["TLE"] + (["TileLang"] if _HAVE_TILELANG else []))
_BENCH_STYLES = ([("red", "-")] + [("orange", "-")] + ([("blue", "-")] if _HAVE_TILELANG else []))
_BENCH_X_VALS = [
    # (B, S, SKV, H, HKV, DQK, DV, topk)
    (1, 512, 1024, 128, 1, 192, 128, 512),
    (1, 1024, 2048, 128, 1, 192, 128, 1024),
    (1, 2048, 4096, 128, 1, 192, 128, 2048),
    (1, 1024, 2048, 128, 1, 160, 128, 1024),
]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "SKV", "H", "HKV", "DQK", "DV", "topk"],
        x_vals=_BENCH_X_VALS,
        x_log=False,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-sparse-mla-fwd",
        args={},
    ))
def benchmark_sparse_mla_fwd(
    B,
    S,
    SKV,
    H,
    HKV,
    DQK,
    DV,
    topk,
    provider,
    warmup,
    rep,
    tilelang_block_I,
    tilelang_num_stages,
    tilelang_threads,
):
    dtype = torch.bfloat16
    q, kv, indices = _build_sparse_mla_inputs(B=B, S=S, SKV=SKV, H=H, HKV=HKV, DQK=DQK, topk=topk, dtype=dtype, seed=1)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":

        def run():
            triton_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    elif provider == "tle":

        def run():
            tle_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    else:
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_interface(
                q,
                kv,
                indices,
                d_v=DV,
                block_I=resolved_block_i,
                num_stages=tilelang_num_stages,
                threads=tilelang_threads,
            )

    try:
        ms, min_ms, max_ms = triton.testing.do_bench(
            run,
            quantiles=quantiles,
            warmup=warmup,
            rep=rep,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime/resource limits
        print(f"[bench:{provider}] failed for "
              f"(B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}, DQK={DQK}, DV={DV}, topk={topk}): {exc}")
        return float("nan"), float("nan"), float("nan")
    return ms, max_ms, min_ms


def run_bench_table(warmup=100, rep=50, show_plots=False, tilelang_block_I=64, tilelang_num_stages=2,
                    tilelang_threads=256):
    benchmark_sparse_mla_fwd.run(
        print_data=True,
        show_plots=show_plots,
        warmup=warmup,
        rep=rep,
        tilelang_block_I=tilelang_block_I,
        tilelang_num_stages=tilelang_num_stages,
        tilelang_threads=tilelang_threads,
    )


def test_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=4096,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    check_tle=True,
    check_tilelang=False,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
):
    q, kv, indices = _build_sparse_mla_inputs(B=B, S=S, SKV=SKV, H=H, HKV=HKV, DQK=DQK, topk=topk, dtype=dtype, seed=0)
    ref_bf16_out = ref_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    triton_bf16_out, triton_bf16_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
    print("triton (no TLE API) bf16 done \n triton lse tensor: \n", triton_bf16_lse)
    print()

    assert torch.allclose(
        triton_bf16_out.float(),
        ref_bf16_out.float(),
        atol=1e-1,
        rtol=1e-1,
    ), "Triton sparse MLA fwd bf16 does not match reference"
    print("Triton sparse MLA fwd bf16 matches reference!")

    if check_tle:
        tle_bf16_out, tle_bf16_lse = tle_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
        print("tle bf16 done \n tle lse tensor: \n", tle_bf16_lse)
        print()
        assert torch.allclose(
            tle_bf16_out.float(),
            ref_bf16_out.float(),
            atol=1e-1,
            rtol=1e-1,
        ), "TLE sparse MLA fwd bf16 does not match reference"
        print("TLE sparse MLA fwd bf16 matches reference!")

    if check_tilelang:
        if not _HAVE_TILELANG:
            raise RuntimeError("TileLang is not installed, cannot run TileLang correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        tilelang_bf16_out, _tilelang_bf16_lse = tilelang_sparse_mla_fwd_interface(
            q,
            kv,
            indices,
            d_v=DV,
            block_I=resolved_block_i,
            num_stages=tilelang_num_stages,
            threads=tilelang_threads,
        )
        assert torch.allclose(
            tilelang_bf16_out.float(),
            ref_bf16_out.float(),
            atol=1e-1,
            rtol=1e-1,
        ), "TileLang sparse MLA fwd bf16 does not match reference"
        print("TileLang sparse MLA fwd bf16 matches reference!")


def bench_sparse_mla_fwd(
    B=1,
    S=4096,
    SKV=4096,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    warmup=250,
    rep=100,
    check_outputs=True,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
):
    q, kv, indices = _build_sparse_mla_inputs(B=B, S=S, SKV=SKV, H=H, HKV=HKV, DQK=DQK, topk=topk, dtype=dtype, seed=0)
    results = []

    def run_triton():
        return triton_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    triton_out, _ = run_triton()
    triton_ms = _bench_ms(run_triton, warmup=warmup, rep=rep)
    triton_tflops = _sparse_mla_tflops(B, S, H, DQK, DV, topk, triton_ms)
    results.append(("triton", triton_ms, triton_tflops))

    tle_out = None
    tilelang_out = None

    def run_tle():
        return tle_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    try:
        tle_out, _ = run_tle()
        tle_ms = _bench_ms(run_tle, warmup=warmup, rep=rep)
        tle_tflops = _sparse_mla_tflops(B, S, H, DQK, DV, topk, tle_ms)
        results.append(("tle", tle_ms, tle_tflops))
    except Exception as exc:  # pragma: no cover - depends on tle/runtime constraints
        print(f"TLE bench skipped due to compile/runtime error: {exc}")

    if _HAVE_TILELANG:
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        if resolved_block_i != tilelang_block_I:
            print(f"TileLang block_I auto-adjusted from {tilelang_block_I} to {resolved_block_i} "
                  f"for topk={topk}.")

        def run_tilelang():
            return tilelang_sparse_mla_fwd_interface(
                q,
                kv,
                indices,
                d_v=DV,
                block_I=resolved_block_i,
                num_stages=tilelang_num_stages,
                threads=tilelang_threads,
            )

        try:
            tilelang_out, _ = run_tilelang()
            tilelang_ms = _bench_ms(run_tilelang, warmup=warmup, rep=rep)
            tilelang_tflops = _sparse_mla_tflops(B, S, H, DQK, DV, topk, tilelang_ms)
            results.append(("tilelang", tilelang_ms, tilelang_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"TileLang bench skipped due to compile/runtime error: {exc}")
    else:
        print("TileLang is not installed, skip TileLang bench.")

    print(f"{'provider':<18}{'ms':>10}{'tflops':>12}{'speedup':>12}")
    for name, ms, tflops in results:
        print(f"{name:<18}{ms:>10.3f}{tflops:>12.2f}{(triton_ms / ms):>12.2f}x")

    if check_outputs:
        if tle_out is not None:
            assert torch.allclose(
                triton_out.float(),
                tle_out.float(),
                atol=1e-1,
                rtol=1e-1,
            ), "Triton output does not match TLE output"
            print("Triton and TLE outputs match.")
        if tilelang_out is not None:
            assert torch.allclose(
                triton_out.float(),
                tilelang_out.float(),
                atol=1e-1,
                rtol=1e-1,
            ), "Triton output does not match TileLang output"
            print("Triton and TileLang outputs match.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "bench", "bench-single"], default="bench")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=128)
    parser.add_argument("--SKV", type=int, default=1024)
    parser.add_argument("--H", type=int, default=32)
    parser.add_argument("--HKV", type=int, default=1)
    parser.add_argument("--DQK", type=int, default=288)
    parser.add_argument("--DV", type=int, default=256)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--skip-output-check", action="store_true")
    parser.add_argument("--skip-tle-check", action="store_true")
    parser.add_argument("--check-tilelang", action="store_true")
    parser.add_argument("--tilelang-block-I", type=int, default=64)
    parser.add_argument("--tilelang-num-stages", type=int, default=2)
    parser.add_argument("--tilelang-threads", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.mode == "test":
        test_sparse_mla_fwd(
            B=args.B,
            S=args.S,
            SKV=args.SKV,
            H=args.H,
            HKV=args.HKV,
            DQK=args.DQK,
            DV=args.DV,
            topk=args.topk,
            dtype=dtype,
            check_tle=not args.skip_tle_check,
            check_tilelang=args.check_tilelang,
            tilelang_block_I=args.tilelang_block_I,
            tilelang_num_stages=args.tilelang_num_stages,
            tilelang_threads=args.tilelang_threads,
        )
    elif args.mode == "bench-single":
        bench_sparse_mla_fwd(
            B=args.B,
            S=args.S,
            SKV=args.SKV,
            H=args.H,
            HKV=args.HKV,
            DQK=args.DQK,
            DV=args.DV,
            topk=args.topk,
            dtype=dtype,
            warmup=args.warmup,
            rep=args.rep,
            check_outputs=not args.skip_output_check,
            tilelang_block_I=args.tilelang_block_I,
            tilelang_num_stages=args.tilelang_num_stages,
            tilelang_threads=args.tilelang_threads,
        )
    else:
        run_bench_table(
            warmup=args.warmup,
            rep=args.rep,
            show_plots=args.show_plots,
            tilelang_block_I=args.tilelang_block_I,
            tilelang_num_stages=args.tilelang_num_stages,
            tilelang_threads=args.tilelang_threads,
        )
