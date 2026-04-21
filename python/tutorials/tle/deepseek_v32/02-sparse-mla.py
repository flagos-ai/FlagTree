# flagtree
"""
Sparse MLA Forward
==================

This tutorial provides:
- Triton sparse MLA forward kernel (no TLE API in kernel body)
- Triton+TLE sparse MLA forward kernel (shared-memory staging)
- optional TileLang sparse MLA forward kernels (baseline and pipelined TileLang examples)
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

try:
    import flash_mla

    _HAVE_FLASHMLA = True
except Exception:  # pragma: no cover - optional dependency
    flash_mla = None
    _HAVE_FLASHMLA = False

TILELANG_SPARSE_MLA_THREADS = 256
TILELANG_SPARSE_MLA_NUM_STAGES = 2
TRITON_SPARSE_MLA_NUM_WARPS = TILELANG_SPARSE_MLA_THREADS // 32
TRITON_SPARSE_MLA_NUM_STAGES = TILELANG_SPARSE_MLA_NUM_STAGES
TLE_SPARSE_MLA_NUM_WARPS = TILELANG_SPARSE_MLA_THREADS // 32
TLE_SPARSE_MLA_NUM_STAGES = TILELANG_SPARSE_MLA_NUM_STAGES


@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    topk_lengths,
    sm_scale: tl.constexpr,
    output,
    lse,
    B,
    SQ,
    SKV,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    VG: tl.constexpr,
    RH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr,
):
    stride_qh: tl.constexpr = TD + D
    stride_qm = H * stride_qh
    stride_qb = SQ * stride_qm
    stride_kvg: tl.constexpr = TD + D
    stride_kvn = VG * stride_kvg
    stride_kvb = SKV * stride_kvn
    stride_tg = K
    stride_tm = VG * stride_tg
    stride_tb = SQ * stride_tm
    stride_oh: tl.constexpr = D
    stride_om = H * stride_oh
    stride_ob = SQ * stride_om
    stride_lm = H
    stride_lb = SQ * stride_lm

    i_b, i_sq, i_gbh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_bh = i_gbh // G, i_gbh % G
    i_b64 = i_b.to(tl.int64)
    i_sq64 = i_sq.to(tl.int64)
    i_g64 = i_g.to(tl.int64)
    i_gbh64 = i_gbh.to(tl.int64)
    q_base = q + i_b64 * stride_qb + i_sq64 * stride_qm + i_gbh64 * (BH * stride_qh)
    tq_base = q_base + D
    kv_base = kv + i_b64 * stride_kvb + i_g64 * stride_kvg
    tkv_base = kv_base + D
    t_base = indices + i_b64 * stride_tb + i_sq64 * stride_tm + i_g64 * stride_tg
    topk_len = tl.load(topk_lengths + i_b64 * (SQ * VG) + i_sq64 * VG + i_g64)
    o_base = output + i_b64 * stride_ob + i_sq64 * stride_om + i_gbh64 * (BH * stride_oh)
    l_base = lse + i_b64 * stride_lb + i_sq64 * stride_lm + i_gbh64 * BH

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = i_bh * BH + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_msk = mask_h[:, None] & mask_d[None, :]
    q_blk = tl.load(q_ptr, q_msk, other=0.0)

    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :]
    tq_msk = mask_h[:, None] & mask_td[None, :]
    tq_blk = tl.load(tq_ptr, tq_msk, other=0.0)

    max_prev = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float32)
    acc = tl.zeros([BH, DP], dtype=tl.float32)

    log_scale: tl.constexpr = sm_scale * 1.44269504
    max_col = i_sq if is_causal else SKV - 1

    NK = tl.cdiv(topk_len, BK)
    for ck in tl.range(NK, num_stages=2):
        t_ptr = BK * ck + offs_t
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)

        kv_ids64 = kv_ids.to(tl.int64)
        kv_ptr = kv_base + kv_ids64[:, None] * stride_kvn + offs_d[None, :]
        kv_msk = mask_ids[:, None] & mask_d[None, :]
        kv_blk = tl.load(kv_ptr, kv_msk, other=0.0)

        tkv_ptr = tkv_base + kv_ids64[:, None] * stride_kvn + offs_td[None, :]
        tkv_msk = mask_ids[:, None] & mask_td[None, :]
        tkv_blk = tl.load(tkv_ptr, tkv_msk, other=0.0)

        qk = tl.full([BH, BK], 0.0, dtype=tl.float32)
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        qk = tl.dot(tq_blk, tl.trans(tkv_blk), qk, out_dtype=tl.float32)
        qk = tl.dot(q_blk, tl.trans(kv_blk), qk, out_dtype=tl.float32)

        new_max = tl.maximum(max_prev, tl.max(qk, axis=1))
        alpha = tl.math.exp2((max_prev - new_max) * log_scale)
        exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)
        sum_qk = tl.sum(exp_qk, axis=1)
        sum_exp = sum_exp * alpha + sum_qk
        acc = acc * alpha[:, None]
        exp_qk = exp_qk.to(tl.bfloat16)
        acc = tl.dot(exp_qk, kv_blk, acc, out_dtype=tl.float32)
        max_prev = new_max

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :]
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_prev * log_scale + tl.math.log2(sum_exp.to(tl.float32))
    l_ptr = l_base + offs_h
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)


@triton.jit
def tle_sparse_mla_fwd(
    q,
    kv,
    indices,
    topk_lengths,
    sm_scale: tl.constexpr,
    output,
    lse,
    B,
    SQ,
    SKV,
    K: tl.constexpr,
    D: tl.constexpr,
    TD: tl.constexpr,
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    VG: tl.constexpr,
    RH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr,
):
    stride_qh: tl.constexpr = TD + D
    stride_qm = H * stride_qh
    stride_qb = SQ * stride_qm
    stride_kvg: tl.constexpr = TD + D
    stride_kvn = VG * stride_kvg
    stride_kvb = SKV * stride_kvn
    stride_tg = K
    stride_tm = VG * stride_tg
    stride_tb = SQ * stride_tm
    stride_oh: tl.constexpr = D
    stride_om = H * stride_oh
    stride_ob = SQ * stride_om
    stride_lm = H
    stride_lb = SQ * stride_lm

    # TileLang-style forward path:
    # - stage Q/Q_tail once in shared memory;
    # - for each sparse K tile, stage KV/K_tail into shared memory;
    # - online softmax on logits;
    # - use probabilities directly for the second GEMM.
    i_b, i_sq, i_grh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_g, i_rh = i_grh // RH, i_grh % RH
    h_base = i_rh * BH
    q_head_base = i_g * G + h_base
    i_b64 = i_b.to(tl.int64)
    i_sq64 = i_sq.to(tl.int64)
    i_g64 = i_g.to(tl.int64)
    q_head_base64 = q_head_base.to(tl.int64)
    q_base = q + i_b64 * stride_qb + i_sq64 * stride_qm + q_head_base64 * stride_qh
    tq_base = q_base + D
    kv_base = kv + i_b64 * stride_kvb + i_g64 * stride_kvg
    tkv_base = kv_base + D
    t_base = indices + i_b64 * stride_tb + i_sq64 * stride_tm + i_g64 * stride_tg
    topk_len = tl.load(topk_lengths + i_b64 * (SQ * VG) + i_sq64 * VG + i_g64)
    o_base = output + i_b64 * stride_ob + i_sq64 * stride_om + q_head_base64 * stride_oh
    l_base = lse + i_b64 * stride_lb + i_sq64 * stride_lm + q_head_base64

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, DP)
    offs_td = tl.arange(0, TDP)
    offs_od = tl.arange(0, DP)
    offs_t = tl.arange(0, BK)
    mask_h = h_base + offs_h < G
    mask_d = offs_d < D
    mask_td = offs_td < TD
    mask_od = mask_d

    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_msk = mask_h[:, None] & mask_d[None, :]
    tq_ptr = tq_base + offs_h[:, None] * stride_qh + offs_td[None, :]
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
    kv_smem = tle.gpu.alloc(
        [BK, DP],
        dtype=q.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    tkv_smem = tle.gpu.alloc(
        [BK, TDP],
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
    kv_smem_ptr = tle.gpu.local_ptr(kv_smem)
    tkv_smem_ptr = tle.gpu.local_ptr(tkv_smem)
    max_prev = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 1.0, dtype=tl.float32)
    acc = tl.zeros([BH, DP], dtype=tl.float32)

    log_scale: tl.constexpr = sm_scale * 1.44269504
    max_col = i_sq if is_causal else SKV - 1

    NK = tl.cdiv(topk_len, BK)
    for ck in tl.range(NK, num_stages=2):
        t_ptr = BK * ck + offs_t
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids <= max_col) & (kv_ids >= 0)
        kv_ids_safe = tl.where(mask_ids, kv_ids, 0)
        kv_ids_safe64 = kv_ids_safe.to(tl.int64)

        kv_ptr = kv_base + kv_ids_safe64[:, None] * stride_kvn + offs_d[None, :]
        kv_msk = mask_ids[:, None] & mask_d[None, :]
        kv_blk = tl.load(kv_ptr, mask=kv_msk, other=0.0)
        tl.store(kv_smem_ptr, kv_blk, mask=kv_msk)

        tkv_ptr = tkv_base + kv_ids_safe64[:, None] * stride_kvn + offs_td[None, :]
        tkv_msk = mask_ids[:, None] & mask_td[None, :]
        tkv_blk = tl.load(tkv_ptr, mask=tkv_msk, other=0.0)
        tl.store(tkv_smem_ptr, tkv_blk, mask=tkv_msk)

        tq_blk = tl.load(tq_smem_ptr)
        q_blk = tl.load(q_smem_ptr)
        tkv_blk = tl.load(tkv_smem_ptr)
        kv_blk = tl.load(kv_smem_ptr)
        qk = tl.full([BH, BK], 0.0, dtype=tl.float32)
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))
        qk = tl.dot(tq_blk, tl.trans(tkv_blk), qk, out_dtype=tl.float32)
        qk = tl.dot(q_blk, tl.trans(kv_blk), qk, out_dtype=tl.float32)

        new_max = tl.maximum(max_prev, tl.max(qk, axis=1))
        alpha = tl.math.exp2((max_prev - new_max) * log_scale)
        exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)
        sum_qk = tl.sum(exp_qk, axis=1)
        sum_exp = sum_exp * alpha + sum_qk
        acc = acc * alpha[:, None]
        prob = exp_qk.to(q.dtype.element_ty)
        acc = tl.dot(prob, kv_blk, acc, out_dtype=tl.float32)
        max_prev = new_max

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :]
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q.dtype.element_ty), o_msk)

    fin_log = max_prev * log_scale + tl.math.log2(sum_exp.to(tl.float32))
    l_ptr = l_base + offs_h
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q.dtype.element_ty), l_msk)


def _compute_topk_length(indices, skv):
    valid_mask = (indices >= 0) & (indices < skv)
    return valid_mask.sum(dim=-1).to(torch.int32).contiguous()


def _sparse_mla_fwd_interface_impl(kernel, q, kv, indices, topk_length=None, sm_scale=None, return_p_sum: bool = False,
                                   d_v=512, bk=32, is_causal=True, extra_kernel_args=(), launch_kwargs=None):
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
    if topk_length is None:
        topk_length = _compute_topk_length(indices, S)
    assert topk_length.shape == (B, SQ, VG)
    topk_length = topk_length.contiguous()
    G = H // VG
    if sm_scale is None:
        sm_scale = DT**-0.5
    if G > 64:
        assert G % 64 == 0, f"TileLang-aligned TLE path requires heads-per-kv-group to be a multiple of 64, but got {G}"
        BH = 64
        RH = G // 64
    else:
        BH = max(triton.next_power_of_2(G), 16)
        RH = 1
    BK = bk
    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float("-inf"), device=q.device, dtype=q.dtype)
    grid = (B, SQ, VG * RH)
    kernel_args = (
        q,
        kv,
        indices,
        topk_length,
        sm_scale,
        output,
        lse,
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
        RH,
        BK,
        BH,
        is_causal,
    ) + tuple(extra_kernel_args)
    if launch_kwargs is None:
        launch_kwargs = {}
    kernel[grid](*kernel_args, **launch_kwargs)
    return output, lse


def triton_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    is_causal=True,
):
    return _sparse_mla_fwd_interface_impl(
        triton_sparse_mla_fwd,
        q,
        kv,
        indices,
        topk_length=topk_length,
        sm_scale=sm_scale,
        return_p_sum=return_p_sum,
        d_v=d_v,
        bk=32,
        is_causal=is_causal,
        launch_kwargs={
            "num_warps": TRITON_SPARSE_MLA_NUM_WARPS,
            "num_stages": TRITON_SPARSE_MLA_NUM_STAGES,
        },
    )


def tle_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    is_causal=True,
):
    return _sparse_mla_fwd_interface_impl(
        tle_sparse_mla_fwd,
        q,
        kv,
        indices,
        topk_length=topk_length,
        sm_scale=sm_scale,
        return_p_sum=return_p_sum,
        d_v=d_v,
        bk=64,
        is_causal=is_causal,
        launch_kwargs={
            "num_warps": TLE_SPARSE_MLA_NUM_WARPS,
            "num_stages": TLE_SPARSE_MLA_NUM_STAGES,
        },
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
        threads=TILELANG_SPARSE_MLA_THREADS,
    ):
        assert dim == tilelang.math.next_power_of_2(dim), f"dim should be power-of-2, but got {dim}"
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim should be power-of-2, but got {tail_dim}"
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
        topk_length_shape = [batch, seq_len, kv_group]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = T.int32
        dtype = T.bfloat16
        accum_dtype = T.float32

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1, "automatic head padding only supports kv_group == 1"

        BI = block_I
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
                 TopkLengths: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
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
                max_kv_i = q_i if is_causal else seq_len_kv - 1
                topk_len = TopkLengths[b_i, s_i, g_i]
                num_k_blocks = T.ceildiv(topk_len, BI)

                H0 = g_i * padded_H + (0 if replicate_h == 1 else (bx % replicate_h) * 64)
                H1 = H0 + H_per_block

                T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                for i_i in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        index = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        mask[bi_i] = (i_i * BI + bi_i < topk_len) and index >= 0 and index <= max_kv_i

                    for bi_i, d_i in T.Parallel(BI, D):
                        index = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        safe_index = T.if_then_else(mask[bi_i], index, 0)
                        KV_shared[bi_i, d_i] = T.if_then_else(mask[bi_i], KV[b_i, safe_index, g_i, d_i], 0)
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        index = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        safe_index = T.if_then_else(mask[bi_i], index, 0)
                        K_tail_shared[bi_i, d_i] = T.if_then_else(mask[bi_i], KV[b_i, safe_index, g_i, D + d_i], 0)

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

    @tilelang.jit(
        out_idx=[-2, -1],
        compile_flags=[
            "-O3",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=-v,--register-usage-level=10",
            "-DNDEBUG",
        ],
    )
    def tilelang_sparse_mla_fwd_pipelined(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        CP0=True,
        block_I=64,
        num_stages=0,
        threads=384,
    ):
        assert dim == tilelang.math.next_power_of_2(dim), f"dim should be power-of-2, but got {dim}"
        assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim should be power-of-2, but got {tail_dim}"
        assert topk % block_I == 0, "topk should be divisible by block_I"
        assert tilelang.cdiv(topk, block_I) % 2 == 0, "pipelined TileLang path requires an even number of K blocks"
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504
        else:
            sm_scale = sm_scale * 1.44269504

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        topk_length_shape = [batch, seq_len, kv_group]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = T.int32
        dtype = T.bfloat16
        accum_dtype = T.float32

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != head_kv:
            assert kv_group == 1, "automatic head padding only supports kv_group == 1"

        BI = block_I
        D = dim
        D_tail = tail_dim
        KV_stride = kv_stride

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
                 TopkLengths: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
                 q_start_index_s: T.Tensor(1, indices_dtype),  # type: ignore
                 Output: T.Tensor(o_shape, dtype),  # type: ignore
                 Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
                 ):
            with T.Kernel((seq_len - KV_stride + 1 if CP0 else seq_len) * replicate_h, batch, kv_group,
                          threads=threads) as (bx, by, bz):
                Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
                K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                is_kv_valid = T.alloc_shared([2, BI], "bool", scope="shared")
                acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                indices_local = T.alloc_var(indices_dtype)

                bar_q = T.alloc_barrier(arrive_count=384)
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                b_i, g_i = by, bz
                s_i = (bx + (KV_stride - 1 if CP0 else 0)) if replicate_h == 1 else (bx // replicate_h +
                                                                                     (KV_stride - 1 if CP0 else 0))
                q_i = q_start_index_s[0] + s_i
                max_kv_i = (q_i + 1 - KV_stride) // KV_stride if is_causal else seq_len_kv - 1
                topk_len = TopkLengths[b_i, s_i, g_i]
                num_k_pairs = T.ceildiv(T.ceildiv(topk_len, BI), 2)
                H0 = g_i * padded_H + (0 if replicate_h == 1 else (bx % replicate_h) * 64)
                H1 = H0 + H_per_block
                tx = T.get_thread_binding()

                # TileLang 0.1.8 does not expose upstream T.tma_copy. Keep the
                # pipelined KV producer path intact and use a synchronous Q copy
                # for compatibility with the local TileLang package.
                T.copy(Q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
                T.copy(Q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
                T.barrier_arrive(bar_q)

                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))
                    T.fill(acc_o_l, 0)
                    T.barrier_wait(bar_q, 0)
                    for i_i in T.serial(num_k_pairs):
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[0, bi_i], 0, -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)
                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[1, bi_i], 0, -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)
                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    for h_i in T.Parallel(H_per_block):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D // 2])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(num_k_pairs):
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != num_k_pairs - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]
                    T.copy(acc_o_r, O_shared_r)
                    T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2:D])

                elif tx >= 256:
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(num_k_pairs):
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            offset = (i_i * 2) * BI + r * 16 + (tx - 256) // 8
                            indices_local = Indices[b_i, s_i, g_i, offset]
                            is_kv_valid[0, r * 16 + (tx - 256) //
                                        8] = offset < topk_len and indices_local >= 0 and indices_local <= max_kv_i
                            if is_kv_valid[0, r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local, g_i,
                                                                  64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_0_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local, g_i,
                                                                  D // 2 + 64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_0[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 + v] = KV[b_i, indices_local, g_i,
                                                                                     D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            offset = (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8
                            indices_local = Indices[b_i, s_i, g_i, offset]
                            is_kv_valid[1, r * 16 + (tx - 256) //
                                        8] = offset < topk_len and indices_local >= 0 and indices_local <= max_kv_i
                            if is_kv_valid[1, r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_1_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local, g_i,
                                                                  64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_1_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, indices_local, g_i,
                                                                  D // 2 + 64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_1[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 + v] = KV[b_i, indices_local, g_i,
                                                                                     D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        return main

    @tilelang.jit(
        out_idx=[-2, -1],
        compile_flags=[
            "-O3",
            "--ptxas-options=-v,--register-usage-level=10",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
    )
    def tilelang_sparse_mla_fwd_seesaw(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        kv_group=1,
        sm_scale=None,
        is_causal=True,
        CP0=True,
        block_I=64,
        num_stages=0,
        threads=384,
    ):
        assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
        assert tail_dim == tilelang.math.next_power_of_2(
            tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
        assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
        if sm_scale is None:
            sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
        else:
            sm_scale = sm_scale * 1.44269504  # log2(e)

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim + tail_dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        topk_length_shape = [batch, seq_len, kv_group]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = "int32"
        dtype = "bfloat16"
        accum_dtype = "float"

        H = head_kv
        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != H:
            assert kv_group == 1, ("here we solve the H padding automatically, other wise you "
                                   "should handle Q copy and Output copy with your mask (when "
                                   "kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would "
                                   "be handled automatically)")
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        assert NI % 2 == 0, "NI should be a multiple of 2"
        D = dim
        D_tail = tail_dim
        KV_stride = kv_stride
        if head_kv > 64:
            assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1

        # Increasing from 32->64 reduces the time spent reading kvcache. If num_query_head = 128
        # and num_kv_head = 1, the same kvcache originally needed to be read 4 times, but now only 2 times
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        @T.prim_func
        def main(Q: T.Tensor(q_shape, dtype),  # type: ignore
                 KV: T.Tensor(kv_shape, dtype),  # type: ignore
                 Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
                 TopkLengths: T.Tensor(topk_length_shape, indices_dtype),  # type: ignore
                 q_start_index_s: T.Tensor(1, indices_dtype),  # type: ignore
                 Output: T.Tensor(o_shape, dtype),  # type: ignore
                 Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
                 ):
            with T.Kernel(
                    # If CP0 is True (i.e., start of sequence), skip the first (KV_stride - 1)
                    # queries that cannot see any KV. Also be careful that seq_len < kv_stride could cause negative grid size
                (max(0, seq_len - kv_stride + 1) if CP0 else seq_len) * REPLICATE_H,
                    batch,
                    kv_group,
                    threads=threads,
            ) as (bx, by, bz):
                Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)

                KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
                K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)

                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r

                # Whether the kv in current BI is visible for this query
                # Producer alternates writing to buf0 and buf1 masks. To avoid the situation
                # where consumer0 is still reading buf0 mask when producer has already started
                # writing buf1 mask, we use two buf masks
                is_kv_valid = T.alloc_shared([2, BI], "bool", scope="shared")

                acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)

                # WG0 computes S0(BI_2*i), WG1 computes S1(BI_2*i+1), shared via shared memory

                # Reuse K_tail_shared for S_shared to save memory when dimensions match
                # Must reuse, otherwise H100 SM's shared mem is insufficient (> 228kb), this is shared mem bound
                S_shared_0 = K_tail_shared_0
                S_shared_1 = K_tail_shared_1

                # WG0 and WG1 exchange local max with each other, compare to compute global max, and rescale their O_L or O_R accordingly
                row_max_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
                row_max_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

                # Used to store sum of exps for even BI and odd BI respectively, which will be summed up for integration later
                row_sum_shared_0 = T.alloc_shared([H_per_block], accum_dtype)
                row_sum_shared_1 = T.alloc_shared([H_per_block], accum_dtype)

                # acc_s, sumexp, m_i each need to be allocated separately for consumer0 and consumer1
                acc_s_0 = T.alloc_fragment([H_per_block, BI], accum_dtype)
                acc_s_1 = T.alloc_fragment([H_per_block, BI], accum_dtype)

                sumexp_0 = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_0 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev_0 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_peer_0 = T.alloc_fragment([H_per_block], accum_dtype)

                sumexp_1 = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_1 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev_1 = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_peer_1 = T.alloc_fragment([H_per_block], accum_dtype)

                bar_q = T.alloc_barrier(arrive_count=384)

                # Producer -> Consumer Barriers
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)  # Prod arrives
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)  # Prod arrives

                # Consumer -> Producer Barriers (Both consumers must arrive)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)

                # Inter-Consumer Barriers (Seesaw Sync)
                bar_stats_0_ready = T.alloc_barrier(arrive_count=128)  # Cons 0 arrives
                bar_stats_1_ready = T.alloc_barrier(arrive_count=128)  # Cons 1 arrives

                bar_S_0_ready = T.alloc_barrier(arrive_count=128)  # Cons 0 arrives
                bar_S_1_ready = T.alloc_barrier(arrive_count=128)  # Cons 1 arrives

                b_i, g_i = by, bz
                # If it's the first chunk, start computing directly from the (kv_stride - 1)-th token
                s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (bx // REPLICATE_H +
                                                                                     (KV_stride - 1 if CP0 else 0))
                q_i = q_start_index_s[0] + s_i
                # Sometimes to reduce kvcache size, we may not store KV for every token, but store
                # KV every KV_stride tokens (usually the last token in the stride window),
                # so the kv range visible to the current query should be [0:max_kv_i]
                max_kv_i = (q_i + 1 - KV_stride) // KV_stride if is_causal else seq_len_kv - 1
                topk_len = TopkLengths[b_i, s_i, g_i]
                num_k_pairs = T.ceildiv(T.ceildiv(topk_len, BI), 2)

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                H1 = H0 + H_per_block

                tx = T.get_thread_binding()

                T.copy(Q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
                T.copy(Q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                # Non-blockingly increment the barrier's internal counter, producer threads can start loading kv ahead of time
                T.barrier_arrive(bar_q)

                if tx >= 256:
                    # producer: prefetch kvcache to shared mem
                    T.set_max_nreg(72, 0)

                    prefetch_indices_0 = T.alloc_fragment([4], indices_dtype)
                    prefetch_indices_1 = T.alloc_fragment([4], indices_dtype)

                    # Prime the Pump! Prefetch indices for iter_0
                    for r in T.serial(4):
                        # This read will cause a long scoreboard stall, but it only happens once before the loop starts
                        prefetch_indices_0[r] = Indices[b_i, s_i, g_i, r * 16 + (tx - 256) // 8]
                        prefetch_indices_1[r] = Indices[b_i, s_i, g_i, BI + r * 16 + (tx - 256) // 8]

                    for i_i in T.serial(num_k_pairs):
                        # Buffer 0
                        # Wait for both KV_shared_0_l and KV_shared_0_r to be done being used

                        T.barrier_wait(bar_k_0_free[0], (i_i & 1))

                        # Block size `BI` is 64, loading is divided into 4 iterations, each processing 16 indices
                        # Producer has 128 threads total, 8 consecutive threads collaborate to load kv for one index
                        for r in T.serial(4):
                            # mitigate long scoreboard stall here
                            offset = (i_i * 2) * BI + r * 16 + (tx - 256) // 8
                            index = prefetch_indices_0[r]
                            is_kv_valid[0, r * 16 +
                                        (tx - 256) // 8] = offset < topk_len and index >= 0 and index <= max_kv_i
                            if is_kv_valid[0, r * 16 + (tx - 256) // 8]:
                                # Here we assume dim = 512, tail_dim = 64
                                with T.attr("default", "async_scope", 1):
                                    # 8 threads collaborate to load one row of KV_dim (length 512), divided into 4 iterations
                                    # In each iteration, each thread loads 8 consecutive elements for both KV_shared_0_l
                                    # and KV_shared_0_r, 8 threads load 64 elements total for each
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            # (tx - 256) // 8 determines which row the thread is responsible for,
                                            # (tx - 256) % 8 determines which part of the row the thread loads
                                            KV_shared_0_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, index, g_i, 64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_0_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, index, g_i,
                                                                  D // 2 + 64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    # tail_dim (length 64) only needs 8 threads collaborating in one iteration to complete loading
                                    for v in T.vectorized(8):
                                        K_tail_shared_0[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 + v] = KV[b_i, index, g_i,
                                                                                     D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        if i_i + 1 < num_k_pairs:
                            # Async prefetch indices needed for the next round of kv data loading, overlaps with current round to hide latency
                            for r in T.serial(4):
                                prefetch_indices_0[r] = Indices[b_i, s_i, g_i,
                                                                ((i_i + 1) * 2) * BI + r * 16 + (tx - 256) // 8]

                        # Buffer 1
                        T.barrier_wait(bar_k_1_free[0], (i_i & 1))

                        for r in T.serial(4):
                            offset = (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8
                            index = prefetch_indices_1[r]
                            is_kv_valid[1, r * 16 +
                                        (tx - 256) // 8] = offset < topk_len and index >= 0 and index <= max_kv_i
                            if is_kv_valid[1, r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_1_l[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, index, g_i, 64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_1_r[r * 16 + (tx - 256) // 8, 64 * u + (tx - 256) % 8 * 8 +
                                                          v] = KV[b_i, index, g_i,
                                                                  D // 2 + 64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_1[r * 16 + (tx - 256) // 8,
                                                        (tx - 256) % 8 * 8 + v] = KV[b_i, index, g_i,
                                                                                     D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

                        if i_i + 1 < num_k_pairs:
                            for r in T.serial(4):
                                prefetch_indices_1[r] = Indices[b_i, s_i, g_i,
                                                                ((i_i + 1) * 2 + 1) * BI + r * 16 + (tx - 256) // 8]

                elif tx < 128:
                    # Check if 384 threads have already arrived at bar_q (phase0 completed),
                    # if not continue waiting, otherwise pass through directly
                    T.barrier_wait(bar_q, 0)

                    # pre-arrive free barriers to indicate buffers are initially free
                    # At the beginning of phase0, tells producer it can load data into both buffers
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_k_1_free[0])

                    # Consumer 0 (WG0): Responsible for Even Blocks and O_L (Left Half)
                    T.set_max_nreg(216, 1)
                    T.fill(sumexp_0, 0)
                    for h_i in T.Parallel(H_per_block):
                        m_i_0[h_i] = -5e4
                    T.fill(acc_o_l, 0)

                    # Each iteration, two consumers cooperate to compute two BIs
                    for i_i in T.serial(num_k_pairs):
                        # --- Step 1: Compute S0 = Q @ K0^T (Even Block) ---
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        T.fill(acc_s_0, 0)
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s_0, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s_0, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s_0, transpose_B=True, wg_wait=-1)

                        T.copy(m_i_0, m_i_prev_0)
                        T.wait_wgmma(0)

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            if not is_kv_valid[0, bi_i]:
                                acc_s_0[h_i, bi_i] = -5e4
                        T.reduce_max(acc_s_0, m_i_0, dim=1, clear=False)

                        # --- Step 2: Local Softmax Stats & Exchange ---
                        T.copy(m_i_0, row_max_shared_0)
                        T.barrier_arrive(bar_stats_0_ready)
                        # If consumer0 has received the local max from consumer1 at iter_i, this also means
                        # consumer1 has finished using S_0 passed by consumer0 at iter_i-1,
                        # so we can write to it directly without blocking below
                        T.barrier_wait(bar_stats_1_ready, (i_i & 1))
                        T.copy(row_max_shared_1, m_i_peer_0)

                        # Update global max and scale O
                        for h_i in T.Parallel(H_per_block):
                            m_i_0[h_i] = T.max(m_i_0[h_i], m_i_peer_0[h_i])

                        # Scale O_L
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)

                        # Scale SumExp
                        for h_i in T.Parallel(H_per_block):
                            sumexp_0[h_i] *= T.exp2((m_i_prev_0[h_i] - m_i_0[h_i]) * sm_scale)

                        # Compute P0 = exp(S0 - m_new)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s_0[h_i, bi_i] = T.exp2(acc_s_0[h_i, bi_i] * sm_scale - m_i_0[h_i] * sm_scale)

                        # Update SumExp with P0
                        T.reduce_sum(acc_s_0, sumexp_i_0, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp_0[h_i] += sumexp_i_0[h_i]

                        # --- Step 3: O_L += P0 @ V0_L (Self-Attention) ---
                        # Wait for S0 buffer to be free (consumed by peer in prev iter)
                        # T.barrier_wait(bar_S_0_free, (i_i & 1))
                        T.copy(acc_s_0, S_shared_0)
                        T.barrier_arrive(bar_S_0_ready)

                        T.gemm(S_shared_0, KV_shared_0_l, acc_o_l, transpose_B=False, wg_wait=-1)

                        # --- Step 4: O_L += P1 @ V1_L (Cross-Attention) ---
                        # Wait for P1 (S1) from peer
                        T.barrier_wait(bar_S_1_ready, (i_i & 1))

                        T.gemm(S_shared_1, KV_shared_1_l, acc_o_l, transpose_B=False, wg_wait=-1)

                        # NOTE: However, k_0 and k_1 are used by both consumer0 and consumer1, so this doesn't bring much performance improvement
                        # Except for the most recent async gemm (i.e., S_shared_1 @ KV_shared_1_k), all others need to wait to finish
                        T.wait_wgmma(1)
                        T.barrier_arrive(bar_k_0_free[0])
                        # Wait for all async gemms to finish
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_k_1_free[0])

                    T.copy(sumexp_0, row_sum_shared_0)
                    T.barrier_arrive(bar_stats_0_ready)  # Reuse barrier
                    T.barrier_wait(bar_stats_1_ready, num_k_pairs & 1)
                    T.copy(row_sum_shared_1, sumexp_i_0)  # Reuse sumexp_i buffer

                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] += sumexp_i_0[h_i]

                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] /= sumexp_0[h_i]

                    for h_i in T.Parallel(H_per_block):
                        sumexp_0[h_i] = T.log2(sumexp_0[h_i]) + m_i_0[h_i] * sm_scale

                    T.copy(acc_o_l, O_shared_l)
                    T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D // 2])
                    T.copy(sumexp_0, Lse[b_i, s_i, H0:H1])  # Write LSE

                elif tx >= 128 and tx < 256:
                    T.barrier_wait(bar_q, 0)

                    # pre-arrive free barriers to indicate buffers are initially free
                    # At the beginning of phase0, tells producer it can load data into both buffers
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_k_1_free[0])

                    # Consumer 1 (WG1): Responsible for Odd Blocks and O_R (Right Half)
                    # NOTE: 256 * 216 + 128 * 72 = 64,512 < 65536 (H100 SM RegFile Limit),
                    # setting more registers will cause a hang, all values must be multiples of 8
                    T.set_max_nreg(216, 1)
                    T.fill(sumexp_1, 0)
                    for h_i in T.Parallel(H_per_block):
                        m_i_1[h_i] = -5e4
                    T.fill(acc_o_r, 0)

                    for i_i in T.serial(num_k_pairs):
                        # --- Step 1: Compute S1 = Q @ K1^T (Odd Block) ---
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        T.fill(acc_s_1, 0)
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s_1, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s_1, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s_1, transpose_B=True, wg_wait=-1)

                        # --- Step 2: Local Softmax Stats & Exchange ---
                        T.copy(m_i_1, m_i_prev_1)
                        T.wait_wgmma(0)

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            if not is_kv_valid[1, bi_i]:
                                acc_s_1[h_i, bi_i] = -5e4

                        T.reduce_max(acc_s_1, m_i_1, dim=1, clear=False)
                        T.copy(m_i_1, row_max_shared_1)
                        T.barrier_arrive(bar_stats_1_ready)
                        T.barrier_wait(bar_stats_0_ready, (i_i & 1))
                        T.copy(row_max_shared_0, m_i_peer_1)

                        for h_i in T.Parallel(H_per_block):
                            m_i_1[h_i] = T.max(m_i_1[h_i], m_i_peer_1[h_i])

                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)

                        for h_i in T.Parallel(H_per_block):
                            sumexp_1[h_i] *= T.exp2((m_i_prev_1[h_i] - m_i_1[h_i]) * sm_scale)

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s_1[h_i, bi_i] = T.exp2(acc_s_1[h_i, bi_i] * sm_scale - m_i_1[h_i] * sm_scale)

                        T.reduce_sum(acc_s_1, sumexp_i_1, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp_1[h_i] += sumexp_i_1[h_i]

                        # --- Step 3: O_R += P1 @ V1_R (Self-Attention) ---
                        T.copy(acc_s_1, S_shared_1)

                        T.barrier_arrive(bar_S_1_ready)

                        T.gemm(S_shared_1, KV_shared_1_r, acc_o_r, transpose_B=False, wg_wait=-1)

                        # --- Step 4: O_R += P0 @ V0_R (Cross-Attention) ---
                        T.barrier_wait(bar_S_0_ready, (i_i & 1))

                        T.gemm(S_shared_0, KV_shared_0_r, acc_o_r, transpose_B=False, wg_wait=-1)

                        T.wait_wgmma(1)
                        T.barrier_arrive(bar_k_1_free[0])
                        T.wait_wgmma(0)
                        T.barrier_arrive(bar_k_0_free[0])

                    T.copy(sumexp_1, row_sum_shared_1)
                    T.barrier_arrive(bar_stats_1_ready)
                    T.barrier_wait(bar_stats_0_ready, num_k_pairs & 1)
                    T.copy(row_sum_shared_0, sumexp_i_1)

                    for h_i in T.Parallel(H_per_block):
                        sumexp_1[h_i] += sumexp_i_1[h_i]

                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] /= sumexp_1[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2:D])

        return main

else:
    tilelang_sparse_mla_fwd = None
    tilelang_sparse_mla_fwd_pipelined = None
    tilelang_sparse_mla_fwd_seesaw = None


def tilelang_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    if not _HAVE_TILELANG or tilelang_sparse_mla_fwd is None:
        raise RuntimeError("TileLang is not installed, cannot run TileLang sparse MLA bench")

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
    if topk_length is None:
        topk_length = _compute_topk_length(indices, _seq_len_kv)
    assert topk_length.shape == (batch, seq_len, kv_group)
    topk_length = topk_length.contiguous()

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
    out, lse = kernel(q, kv, indices, topk_length)
    return out, lse


def tilelang_sparse_mla_fwd_pipelined_interface(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    is_causal=True,
    block_I=64,
    threads=384,
    q_start_index=1024,
    kv_stride=1,
):
    if not _HAVE_TILELANG or tilelang_sparse_mla_fwd_pipelined is None:
        raise RuntimeError("TileLang is not installed, cannot run pipelined TileLang sparse MLA bench")

    cp0 = is_causal and q_start_index == 0
    assert not return_p_sum, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert dim == triton.next_power_of_2(dim), f"d_v should be power-of-2 for TileLang path, but got {dim}"
    assert tail_dim == triton.next_power_of_2(
        tail_dim), f"tail dim should be power-of-2 for TileLang path, but got {tail_dim}"
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)
    if topk_length is None:
        topk_length = _compute_topk_length(indices, seq_len_kv)
    assert topk_length.shape == (batch, seq_len, kv_group)
    topk_length = topk_length.contiguous()
    resolved_block_i = _resolve_tilelang_block_i(topk, block_I)
    ni = triton.cdiv(topk, resolved_block_i)
    if ni % 2 != 0:
        raise ValueError(
            f"pipelined TileLang path requires an even number of K blocks, got topk={topk}, block_I={resolved_block_i}")

    kernel = tilelang_sparse_mla_fwd_pipelined(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        kv_group,
        sm_scale,
        is_causal,
        cp0,
        block_I=resolved_block_i,
        threads=threads,
    )
    q_start_index_s = torch.tensor([q_start_index], dtype=torch.int32, device=q.device)
    out, lse = kernel(q, kv, indices, topk_length, q_start_index_s)
    if is_causal and q_start_index == 0 and kv_stride > 1:
        out[:, :kv_stride - 1, :, :] = 0
    return out, lse


def tilelang_sparse_mla_fwd_seesaw_interface(
    q,
    kv,
    indices,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    is_causal=True,
    block_I=64,
    threads=384,
    q_start_index=1024,
    kv_stride=1,
):
    if not _HAVE_TILELANG or tilelang_sparse_mla_fwd_seesaw is None:
        raise RuntimeError("TileLang is not installed, cannot run seesaw TileLang sparse MLA bench")

    cp0 = is_causal and q_start_index == 0
    assert not return_p_sum, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert dim == triton.next_power_of_2(dim), f"d_v should be power-of-2 for TileLang path, but got {dim}"
    assert tail_dim == triton.next_power_of_2(
        tail_dim), f"tail dim should be power-of-2 for TileLang path, but got {tail_dim}"
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)
    if topk_length is None:
        topk_length = _compute_topk_length(indices, seq_len_kv)
    assert topk_length.shape == (batch, seq_len, kv_group)
    topk_length = topk_length.contiguous()
    resolved_block_i = _resolve_tilelang_block_i(topk, block_I)
    ni = triton.cdiv(topk, resolved_block_i)
    if ni % 2 != 0:
        raise ValueError(
            f"seesaw TileLang path requires an even number of K blocks, got topk={topk}, block_I={resolved_block_i}")

    kernel = tilelang_sparse_mla_fwd_seesaw(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        tail_dim,
        topk,
        kv_stride,
        kv_group,
        sm_scale,
        is_causal,
        cp0,
        block_I=resolved_block_i,
        threads=threads,
    )
    q_start_index_s = torch.tensor([q_start_index], dtype=torch.int32, device=q.device)
    out, lse = kernel(q, kv, indices, topk_length, q_start_index_s)
    if is_causal and q_start_index == 0 and kv_stride > 1:
        out[:, :kv_stride - 1, :, :] = 0
    return out, lse


def _prepare_flashmla_sparse_prefill_args(q, kv, indices, topk_length=None, sm_scale=None, d_v=512):
    if not _HAVE_FLASHMLA or flash_mla is None:
        raise RuntimeError("FlashMLA is not installed, cannot run FlashMLA sparse MLA bench")

    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, SKV, HKV, _ = kv.shape
    assert indices.shape[:3] == (B, SQ, HKV)
    if HKV != 1:
        raise ValueError(f"FlashMLA sparse prefill on sm90 requires h_kv == 1, but got {HKV}")
    if d_v != 512:
        raise ValueError(f"FlashMLA sparse prefill only supports d_v == 512, but got {d_v}")
    if DT not in (512, 576):
        raise ValueError(f"FlashMLA sparse prefill only supports d_qk in {{512, 576}}, but got {DT}")
    if sm_scale is None:
        sm_scale = DT**-0.5

    padded_h = triton.cdiv(H, 64) * 64
    if padded_h != H:
        q_padded = torch.zeros((B, SQ, padded_h, DT), device=q.device, dtype=q.dtype)
        q_padded[:, :, :H, :] = q
    else:
        q_padded = q

    topk = indices.shape[-1]
    padded_topk = triton.cdiv(max(topk, 1), 128) * 128
    batch_offsets = torch.arange(B, device=indices.device, dtype=indices.dtype).view(B, 1, 1, 1) * SKV
    valid_mask = (indices >= 0) & (indices < SKV)
    flash_indices = torch.where(valid_mask, indices + batch_offsets, -torch.ones_like(indices))
    if topk_length is None:
        topk_length = valid_mask.sum(dim=-1).to(torch.int32)
    assert topk_length.shape == (B, SQ, HKV)
    topk_length = topk_length.reshape(B * SQ).to(torch.int32).contiguous()
    if padded_topk != topk:
        flash_indices_padded = torch.full((B, SQ, HKV, padded_topk), -1, device=indices.device, dtype=indices.dtype)
        flash_indices_padded[:, :, :, :topk] = flash_indices
        flash_indices = flash_indices_padded

    q_flat = q_padded.reshape(B * SQ, padded_h, DT).contiguous()
    kv_flat = kv.reshape(B * SKV, HKV, DT).contiguous()
    indices_flat = flash_indices.reshape(B * SQ, HKV, padded_topk).contiguous()
    return {
        "q": q_flat,
        "kv": kv_flat,
        "indices": indices_flat,
        "sm_scale": sm_scale,
        "d_v": d_v,
        "topk_length": topk_length,
        "B": B,
        "SQ": SQ,
        "H": H,
        "padded_h": padded_h,
    }


def _run_flashmla_sparse_prefill_prepared(prepared):
    out, _max_logits, lse = flash_mla.flash_mla_sparse_fwd(
        prepared["q"],
        prepared["kv"],
        prepared["indices"],
        sm_scale=prepared["sm_scale"],
        d_v=prepared["d_v"],
        topk_length=prepared["topk_length"],
    )
    return out, lse


def flashmla_sparse_mla_fwd_interface(q, kv, indices, topk_length=None, sm_scale=None, return_p_sum: bool = False,
                                      d_v=512):
    assert not return_p_sum, "This kernel file is for fwd only"
    prepared = _prepare_flashmla_sparse_prefill_args(
        q,
        kv,
        indices,
        topk_length=topk_length,
        sm_scale=sm_scale,
        d_v=d_v,
    )
    out, _max_logits, lse = flash_mla.flash_mla_sparse_fwd(
        prepared["q"],
        prepared["kv"],
        prepared["indices"],
        sm_scale=prepared["sm_scale"],
        d_v=prepared["d_v"],
        topk_length=prepared["topk_length"],
    )
    B = prepared["B"]
    SQ = prepared["SQ"]
    H = prepared["H"]
    padded_h = prepared["padded_h"]
    out = out.reshape(B, SQ, padded_h, d_v)[:, :, :H, :]
    lse = lse.reshape(B, SQ, padded_h)[:, :, :H]
    return out, lse


def flashmla_sparse_mla_decode_interface(
    q,
    k_cache,
    indices_in_kvcache,
    topk_length=None,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    tile_scheduler_metadata=None,
):
    if not _HAVE_FLASHMLA or flash_mla is None:
        raise RuntimeError("FlashMLA is not installed, cannot run FlashMLA sparse decode bench")

    assert not return_p_sum, "This kernel file is for fwd only"
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    if tile_scheduler_metadata is None:
        tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_with_kvcache(
        q,
        k_cache,
        None,
        None,
        d_v,
        tile_scheduler_metadata,
        None,
        sm_scale,
        False,
        True,
        indices_in_kvcache,
        None,
        None,
        None,
        topk_length,
        None,
    )
    return out, lse.transpose(1, 2).contiguous()


def _build_sparse_mla_inputs(B=1, S=4096, SKV=4096, H=128, HKV=1, DQK=576, topk=2048, dtype=torch.bfloat16, seed=0,
                             q_start_index=0, kv_stride=1, scale=1.0):
    torch.random.manual_seed(seed)
    q = (torch.randn((B, S, H, DQK), dtype=dtype, device="cuda") * scale).requires_grad_(True)
    kv = (torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda") * scale).requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                valid_kv = min(max(1, (t + q_start_index) // kv_stride), SKV)
                i_i = torch.randperm(valid_kv)[:topk]
                indices[b, t, h, :len(i_i)] = i_i
    return q, kv, indices


def _build_flashmla_sparse_prefill_inputs(
    B=1,
    S=4096,
    SKV=8192,
    H=128,
    HKV=1,
    DQK=576,
    topk=2048,
    dtype=torch.bfloat16,
    seed=0,
    randperm_chunk=64,
):
    if topk > SKV:
        raise ValueError(f"FlashMLA sparse prefill perf cases require topk <= SKV, got topk={topk}, SKV={SKV}")
    torch.random.manual_seed(seed)

    q_offset = (torch.rand((), device="cuda") - 0.5) / 10
    kv_offset = (torch.rand((), device="cuda") - 0.5) / 10
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda") / 10 + q_offset
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda") / 10 + kv_offset
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    rows = B * S * HKV
    indices_cpu = torch.empty((rows, topk), dtype=torch.int32, device="cpu")
    for start in range(0, rows, randperm_chunk):
        stop = min(start + randperm_chunk, rows)
        rand = torch.rand((stop - start, SKV), dtype=torch.float32, device="cpu")
        indices_cpu[start:stop] = rand.topk(topk, dim=-1, sorted=True).indices.to(torch.int32)
    indices = indices_cpu.view(B, S, HKV, topk).to("cuda", non_blocking=True).contiguous()
    topk_length = torch.full((B, S, HKV), topk, dtype=torch.int32, device="cuda")
    return q.contiguous(), kv.contiguous(), indices, topk_length


def _cast_scale_inv_to_ue8m0(scales_inv):
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _quantize_v32_fp8_sparse_k_cache(input_k_cache):
    assert input_k_cache.shape[-1] == 576
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)
    result = torch.empty((num_blocks, block_size + 1, 656), dtype=torch.float8_e4m3fn,
                         device=input_k_cache.device)[:, :block_size, :]
    result_nope = result[..., :512]
    result_scale = result[..., 512:528].view(torch.float32)
    result_rope = result[..., 528:].view(input_k_cache.dtype)
    result_rope[:] = input_k_cache[..., 512:]

    for tile_idx in range(4):
        lo = tile_idx * 128
        hi = lo + 128
        scales_inv = torch.abs(input_k_cache[..., lo:hi]).max(dim=-1).values.float() / 448.0
        scales_inv = _cast_scale_inv_to_ue8m0(scales_inv)
        result_scale[:, :, tile_idx] = scales_inv
        result_nope[..., lo:hi] = (input_k_cache[..., lo:hi].float() / scales_inv.unsqueeze(-1).float()).to(
            torch.float8_e4m3fn)

    return result.view(num_blocks, block_size, 1, 656)


def _dequantize_v32_fp8_sparse_k_cache(quant_k_cache):
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
    result = torch.empty((num_blocks, block_size, 576), dtype=torch.bfloat16, device=quant_k_cache.device)
    input_nope = quant_k_cache[..., :512]
    input_scale = quant_k_cache[..., 512:528].view(torch.float32)
    input_rope = quant_k_cache[..., 528:].view(torch.bfloat16)
    result[..., 512:] = input_rope

    for tile_idx in range(4):
        lo = tile_idx * 128
        hi = lo + 128
        result[..., lo:hi] = input_nope[..., lo:hi].to(torch.float32) * input_scale[..., tile_idx].unsqueeze(-1)

    return result.view(num_blocks, block_size, 1, 576)


def _randperm_batch_cpu(batch_size, perm_range, perm_size, padding=-1, chunk=64):
    perm_range = perm_range.to(torch.int64).cpu()
    result = torch.empty((batch_size, perm_size), dtype=torch.int32, device="cpu")
    for start in range(0, batch_size, chunk):
        stop = min(start + chunk, batch_size)
        cur_range = perm_range[start:stop]
        cur_max = max(int(cur_range.max().item()), perm_size)
        rand = torch.rand((stop - start, cur_max), dtype=torch.float32, device="cpu")
        rand[torch.arange(cur_max).view(1, cur_max) >= cur_range.view(stop - start, 1)] = float("-inf")
        cur = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        cur[cur >= cur_range.view(stop - start, 1)] = padding
        result[start:stop] = cur
    return result


def _abs_indices_to_indices_in_kvcache(abs_indices, block_table, block_size):
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape
    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0
    batch_offsets = torch.arange(b, dtype=torch.int64, device=abs_indices.device).view(b, 1, 1) * max_blocks_per_seq
    real_block_idxs = block_table.view(-1).index_select(0, (abs_indices.to(torch.int64) // block_size +
                                                            batch_offsets).reshape(-1))
    indices = real_block_idxs.view(b, s_q, topk) * block_size + abs_indices % block_size
    indices[invalid_mask] = -1
    return indices.to(torch.int32)


def _build_flashmla_sparse_decode_inputs(
    B=128,
    S=2,
    SKV=32768,
    H=128,
    HKV=1,
    DQK=576,
    topk=2048,
    dtype=torch.bfloat16,
    seed=0,
    block_size=64,
    is_varlen=True,
):
    if DQK != 576:
        raise ValueError("local FlashMLA decode fixture currently implements the V3.2 FP8 cache layout only")
    if HKV != 1:
        raise ValueError("FlashMLA sparse decode fixture requires h_kv == 1")

    torch.random.manual_seed(seed)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").clamp_(min=-1.0, max=1.0).contiguous()

    cache_seqlens_cpu = torch.full((B, ), SKV, dtype=torch.int32, device="cpu")
    if is_varlen:
        sampled = torch.normal(mean=float(SKV), std=float(SKV) / 2, size=(B, ), device="cpu")
        cache_seqlens_cpu = torch.maximum(sampled.to(torch.int32), torch.full((B, ), S, dtype=torch.int32))

    max_seqlen_alignment = 4 * block_size
    max_seqlen_pad = max(triton.cdiv(int(cache_seqlens_cpu.max().item()), max_seqlen_alignment),
                         1) * max_seqlen_alignment
    max_num_blocks = max_seqlen_pad // block_size

    block_table = torch.arange(B * max_num_blocks, dtype=torch.int32, device="cpu").view(B, max_num_blocks)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(B, max_num_blocks)

    blocked_k = (torch.randn(
        (block_table.numel(), block_size, HKV, DQK), dtype=dtype, device="cuda") / 10).clamp_(min=-1.0, max=1.0)
    abs_indices = _randperm_batch_cpu(B * S, cache_seqlens_cpu.repeat_interleave(S), topk, padding=-1).view(B, S, topk)
    indices_in_kvcache_cpu = _abs_indices_to_indices_in_kvcache(abs_indices, block_table, block_size)
    indices_in_kvcache = indices_in_kvcache_cpu.to("cuda", non_blocking=True).contiguous()

    k_cache_quantized = _quantize_v32_fp8_sparse_k_cache(blocked_k)
    k_cache_dequantized = _dequantize_v32_fp8_sparse_k_cache(k_cache_quantized).contiguous()
    block_table_cuda = block_table.to(device="cuda", dtype=torch.int64, non_blocking=True)
    kv_for_sparsemla = k_cache_dequantized.index_select(0, block_table_cuda.reshape(-1)).view(
        B, max_num_blocks, block_size, HKV, DQK).reshape(B, max_seqlen_pad, HKV, DQK).contiguous()

    q_flat = q.contiguous()
    indices_flat = abs_indices.to("cuda", non_blocking=True).view(B, S, HKV, topk).contiguous()
    topk_length_flat = _compute_topk_length(indices_flat, kv_for_sparsemla.shape[1])

    return {
        "q": q,
        "k_cache_quantized": k_cache_quantized,
        "indices_in_kvcache": indices_in_kvcache,
        "topk_length": None,
        "q_flat": q_flat,
        "kv_flat": kv_for_sparsemla,
        "indices_flat": indices_flat,
        "topk_length_flat": topk_length_flat,
        "sm_scale": DQK**-0.55,
    }


def _resolve_tilelang_block_i(topk: int, block_i: int) -> int:
    if block_i <= 0:
        raise ValueError(f"tilelang block_I should be > 0, but got {block_i}")
    if topk % block_i == 0:
        return block_i
    fallback = math.gcd(topk, block_i)
    if fallback <= 0:
        raise ValueError(f"cannot find a valid tilelang block_I for topk={topk}, block_I={block_i}")
    return fallback


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=512, q_start_index=0, kv_stride=1):
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
    if is_casual:
        compressed_casual_mask = torch.arange(q_start_index, sq + q_start_index, dtype=torch.int32, device="cuda").view(
            -1, 1) >= torch.arange(kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32,
                                   device="cuda").view(1, -1)
    else:
        compressed_casual_mask = torch.ones((sq, sk), dtype=torch.bool, device="cuda")

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    if is_casual:
        mask[:, :, :kv_stride - 1, 0] = True
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


def _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, ms):
    total_topk = int(topk_length.to(torch.int64).sum().item())
    return (total_topk * (DQK + DV) * 2 * H) / (ms * 1e-3) / 1e12


BENCH_DEFAULT_WARMUP_MS = 2000
BENCH_DEFAULT_REP_MS = 5000
BENCH_DEFAULT_SEED = 1


def _bench_ms(fn, warmup=BENCH_DEFAULT_WARMUP_MS, rep=BENCH_DEFAULT_REP_MS):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return float(ms if not isinstance(ms, tuple) else ms[0])


_BENCH_PROVIDERS = (["triton", "tle"] +
                    (["tilelang", "tilelang-pipelined", "tilelang-seesaw"] if _HAVE_TILELANG else []) +
                    (["flashmla"] if _HAVE_FLASHMLA else []))
_BENCH_NAMES = (["Triton", "TLE"] + (["TileLang", "TileLang-Pipelined", "TileLang-Seesaw"] if _HAVE_TILELANG else []) +
                (["FlashMLA"] if _HAVE_FLASHMLA else []))
_BENCH_STYLES = ([("red", "-"), ("orange", "-")] + ([("blue", "-"), ("cyan", "-"),
                                                     ("purple", "-")] if _HAVE_TILELANG else []) +
                 ([("green", "-")] if _HAVE_FLASHMLA else []))
_BENCH_X_VALS = [
    # FlashMLA v0.1.8 sparse prefill V3.2 performance cases:
    # TestParam(s_q=4096, s_kv, topk=2048, h_q=128, h_kv=1, d_qk=576, d_v=512).
    # attn_sink is intentionally omitted here because the local Triton/TLE/TileLang kernels do not implement it.
    (1, 4096, 8192, 128, 1, 576, 512, 2048),
    (1, 4096, 32768, 128, 1, 576, 512, 2048),
    (1, 4096, 65536, 128, 1, 576, 512, 2048),
    (1, 4096, 98304, 128, 1, 576, 512, 2048),
    (1, 4096, 131072, 128, 1, 576, 512, 2048),
]

_BENCH_INPUT_CACHE = {"key": None, "value": None}
_DECODE_BENCH_X_VALS = [
    # FlashMLA v0.1.8 sparse decoding V3.2 production cases:
    # RawTestParam(b, h_q=128, s_q=2, h_kv=1, s_kv=32768, is_varlen=True, topk=2048, d_qk=576).
    (2, 2, 32768, 128, 1, 576, 512, 2048, 64),
    (64, 2, 32768, 128, 1, 576, 512, 2048, 64),
    (74, 2, 32768, 128, 1, 576, 512, 2048, 64),
    (128, 2, 32768, 128, 1, 576, 512, 2048, 64),
]
_DECODE_BENCH_INPUT_CACHE = {"key": None, "value": None}


def _get_bench_sparse_mla_inputs(B, S, SKV, H, HKV, DQK, topk, dtype, seed, input_mode):
    key = (B, S, SKV, H, HKV, DQK, topk, dtype, seed, input_mode)
    if _BENCH_INPUT_CACHE["key"] != key:
        if input_mode == "flashmla":
            _BENCH_INPUT_CACHE["value"] = _build_flashmla_sparse_prefill_inputs(
                B=B,
                S=S,
                SKV=SKV,
                H=H,
                HKV=HKV,
                DQK=DQK,
                topk=topk,
                dtype=dtype,
                seed=seed,
            )
        elif input_mode == "causal":
            q, kv, indices = _build_sparse_mla_inputs(
                B=B,
                S=S,
                SKV=SKV,
                H=H,
                HKV=HKV,
                DQK=DQK,
                topk=topk,
                dtype=dtype,
                seed=seed,
            )
            _BENCH_INPUT_CACHE["value"] = (q, kv, indices, _compute_topk_length(indices, SKV))
        else:
            raise ValueError(f"unknown sparse MLA bench input mode: {input_mode}")
        _BENCH_INPUT_CACHE["key"] = key
    return _BENCH_INPUT_CACHE["value"]


def _get_decode_bench_sparse_mla_inputs(B, S, SKV, H, HKV, DQK, topk, dtype, seed, block_size):
    key = (B, S, SKV, H, HKV, DQK, topk, dtype, seed, block_size)
    if _DECODE_BENCH_INPUT_CACHE["key"] != key:
        _DECODE_BENCH_INPUT_CACHE["value"] = _build_flashmla_sparse_decode_inputs(
            B=B,
            S=S,
            SKV=SKV,
            H=H,
            HKV=HKV,
            DQK=DQK,
            topk=topk,
            dtype=dtype,
            seed=seed,
            block_size=block_size,
            is_varlen=True,
        )
        _DECODE_BENCH_INPUT_CACHE["key"] = key
    return _DECODE_BENCH_INPUT_CACHE["value"]


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
    input_mode,
    seed,
):
    dtype = torch.bfloat16
    is_causal = input_mode == "causal"
    sm_scale = 0.5 if input_mode == "flashmla" else None
    q, kv, indices, topk_length = _get_bench_sparse_mla_inputs(
        B,
        S,
        SKV,
        H,
        HKV,
        DQK,
        topk,
        dtype,
        seed=seed,
        input_mode=input_mode,
    )
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":

        def run():
            triton_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, sm_scale=sm_scale, d_v=DV,
                                            is_causal=is_causal)

    elif provider == "tle":

        def run():
            tle_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, sm_scale=sm_scale, d_v=DV,
                                         is_causal=is_causal)

    elif provider == "tilelang-pipelined":
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_pipelined_interface(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
                block_I=resolved_block_i,
                threads=384,
                q_start_index=0,
                kv_stride=1,
            )

    elif provider == "tilelang-seesaw":
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_seesaw_interface(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
                block_I=resolved_block_i,
                threads=384,
                q_start_index=0,
                kv_stride=1,
            )

    elif provider == "flashmla":
        if not _HAVE_FLASHMLA:
            return float("nan"), float("nan"), float("nan")
        prepared_flashmla = _prepare_flashmla_sparse_prefill_args(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=sm_scale,
            d_v=DV,
        )

        def run():
            _run_flashmla_sparse_prefill_prepared(prepared_flashmla)

    else:
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_interface(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
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


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["B", "S", "SKV", "H", "HKV", "DQK", "DV", "topk", "block_size"],
        x_vals=_DECODE_BENCH_X_VALS,
        x_log=False,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-sparse-mla-decode",
        args={},
    ))
def benchmark_sparse_mla_decode(
    B,
    S,
    SKV,
    H,
    HKV,
    DQK,
    DV,
    topk,
    block_size,
    provider,
    warmup,
    rep,
    tilelang_block_I,
    tilelang_num_stages,
    tilelang_threads,
):
    dtype = torch.bfloat16
    inputs = _get_decode_bench_sparse_mla_inputs(
        B,
        S,
        SKV,
        H,
        HKV,
        DQK,
        topk,
        dtype,
        seed=2,
        block_size=block_size,
    )
    sm_scale = inputs["sm_scale"]
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":

        def run():
            triton_sparse_mla_fwd_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
            )

    elif provider == "tle":

        def run():
            tle_sparse_mla_fwd_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
            )

    elif provider == "tilelang-pipelined":
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_pipelined_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
                block_I=resolved_block_i,
                threads=384,
                q_start_index=0,
                kv_stride=1,
            )

    elif provider == "tilelang-seesaw":
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_seesaw_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
                block_I=resolved_block_i,
                threads=384,
                q_start_index=0,
                kv_stride=1,
            )

    elif provider == "flashmla":
        if not _HAVE_FLASHMLA:
            return float("nan"), float("nan"), float("nan")
        tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

        def run():
            flashmla_sparse_mla_decode_interface(
                inputs["q"],
                inputs["k_cache_quantized"],
                inputs["indices_in_kvcache"],
                topk_length=inputs["topk_length"],
                sm_scale=sm_scale,
                d_v=DV,
                tile_scheduler_metadata=tile_scheduler_metadata,
            )

    else:
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run():
            tilelang_sparse_mla_fwd_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
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
        print(f"[decode-bench:{provider}] failed for "
              f"(B={B}, S={S}, SKV={SKV}, H={H}, HKV={HKV}, DQK={DQK}, DV={DV}, topk={topk}): {exc}")
        return float("nan"), float("nan"), float("nan")
    return ms, max_ms, min_ms


def run_bench_table(warmup=BENCH_DEFAULT_WARMUP_MS, rep=BENCH_DEFAULT_REP_MS, show_plots=False, tilelang_block_I=64,
                    tilelang_num_stages=2, tilelang_threads=256, input_mode="flashmla", seed=BENCH_DEFAULT_SEED):
    benchmark_sparse_mla_fwd.run(
        print_data=True,
        show_plots=show_plots,
        warmup=warmup,
        rep=rep,
        tilelang_block_I=tilelang_block_I,
        tilelang_num_stages=tilelang_num_stages,
        tilelang_threads=tilelang_threads,
        input_mode=input_mode,
        seed=seed,
    )


def run_decode_bench_table(warmup=BENCH_DEFAULT_WARMUP_MS, rep=BENCH_DEFAULT_REP_MS, show_plots=False,
                           tilelang_block_I=64, tilelang_num_stages=2, tilelang_threads=256):
    benchmark_sparse_mla_decode.run(
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
    check_tilelang_pipelined=False,
    check_tilelang_seesaw=False,
    check_flashmla=False,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
):
    q, kv, indices = _build_sparse_mla_inputs(B=B, S=S, SKV=SKV, H=H, HKV=HKV, DQK=DQK, topk=topk, dtype=dtype, seed=0)
    topk_length = _compute_topk_length(indices, SKV)
    ref_bf16_out = ref_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    triton_bf16_out, triton_bf16_lse = triton_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, d_v=DV)
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
        tle_bf16_out, tle_bf16_lse = tle_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, d_v=DV)
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
            topk_length=topk_length,
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

    if check_tilelang_pipelined:
        if not _HAVE_TILELANG:
            raise RuntimeError("TileLang is not installed, cannot run pipelined TileLang correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        q_pipelined, kv_pipelined, indices_pipelined = _build_sparse_mla_inputs(
            B=B,
            S=S,
            SKV=SKV,
            H=H,
            HKV=HKV,
            DQK=DQK,
            topk=topk,
            dtype=dtype,
            seed=0,
            q_start_index=1024,
            kv_stride=1,
            scale=0.1,
        )
        topk_length_pipelined = _compute_topk_length(indices_pipelined, SKV)
        ref_pipelined_bf16_out = ref_sparse_mla_fwd_interface(
            q_pipelined,
            kv_pipelined,
            indices_pipelined,
            d_v=DV,
            q_start_index=1024,
            kv_stride=1,
        )
        tilelang_pipelined_bf16_out, _tilelang_pipelined_bf16_lse = tilelang_sparse_mla_fwd_pipelined_interface(
            q_pipelined,
            kv_pipelined,
            indices_pipelined,
            topk_length=topk_length_pipelined,
            d_v=DV,
            block_I=resolved_block_i,
            q_start_index=1024,
            kv_stride=1,
        )
        assert torch.allclose(
            tilelang_pipelined_bf16_out.float(),
            ref_pipelined_bf16_out.float(),
            atol=1e-3,
            rtol=1e-3,
        ), "Pipelined TileLang sparse MLA fwd bf16 does not match reference"
        print("Pipelined TileLang sparse MLA fwd bf16 matches reference!")

    if check_tilelang_seesaw:
        if not _HAVE_TILELANG:
            raise RuntimeError("TileLang is not installed, cannot run seesaw TileLang correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        q_seesaw, kv_seesaw, indices_seesaw = _build_sparse_mla_inputs(
            B=B,
            S=S,
            SKV=SKV,
            H=H,
            HKV=HKV,
            DQK=DQK,
            topk=topk,
            dtype=dtype,
            seed=0,
            q_start_index=1024,
            kv_stride=1,
            scale=0.1,
        )
        topk_length_seesaw = _compute_topk_length(indices_seesaw, SKV)
        ref_seesaw_bf16_out = ref_sparse_mla_fwd_interface(
            q_seesaw,
            kv_seesaw,
            indices_seesaw,
            d_v=DV,
            q_start_index=1024,
            kv_stride=1,
        )
        tilelang_seesaw_bf16_out, _tilelang_seesaw_bf16_lse = tilelang_sparse_mla_fwd_seesaw_interface(
            q_seesaw,
            kv_seesaw,
            indices_seesaw,
            topk_length=topk_length_seesaw,
            d_v=DV,
            block_I=resolved_block_i,
            q_start_index=1024,
            kv_stride=1,
        )
        assert torch.allclose(
            tilelang_seesaw_bf16_out.float(),
            ref_seesaw_bf16_out.float(),
            atol=1e-3,
            rtol=1e-3,
        ), "Seesaw TileLang sparse MLA fwd bf16 does not match reference"
        print("Seesaw TileLang sparse MLA fwd bf16 matches reference!")

    if check_flashmla:
        if not _HAVE_FLASHMLA:
            raise RuntimeError("FlashMLA is not installed, cannot run FlashMLA correctness check")
        flashmla_bf16_out, _flashmla_bf16_lse = flashmla_sparse_mla_fwd_interface(q, kv, indices,
                                                                                  topk_length=topk_length, d_v=DV)
        assert torch.allclose(
            flashmla_bf16_out.float(),
            ref_bf16_out.float(),
            atol=1e-1,
            rtol=1e-1,
        ), "FlashMLA sparse MLA fwd bf16 does not match reference"
        print("FlashMLA sparse MLA fwd bf16 matches reference!")


def test_sparse_mla_decode(
    B=4,
    S=2,
    SKV=512,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=64,
    dtype=torch.bfloat16,
    check_tle=True,
    check_tilelang=False,
    check_tilelang_pipelined=False,
    check_tilelang_seesaw=False,
    check_flashmla=True,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
):
    inputs = _build_flashmla_sparse_decode_inputs(
        B=B,
        S=S,
        SKV=SKV,
        H=H,
        HKV=HKV,
        DQK=DQK,
        topk=topk,
        dtype=dtype,
        seed=3,
        block_size=tilelang_block_I,
        is_varlen=True,
    )
    sm_scale = inputs["sm_scale"]

    triton_out_flat, _ = triton_sparse_mla_fwd_interface(
        inputs["q_flat"],
        inputs["kv_flat"],
        inputs["indices_flat"],
        topk_length=inputs["topk_length_flat"],
        sm_scale=sm_scale,
        d_v=DV,
        is_causal=False,
    )
    triton_out = triton_out_flat.view(B, S, H, DV)
    print("triton sparse decode-compatible path bf16 done")

    ref_out = triton_out
    if check_flashmla:
        if not _HAVE_FLASHMLA:
            raise RuntimeError("FlashMLA is not installed, cannot run FlashMLA sparse decode correctness check")
        flashmla_out, _flashmla_lse = flashmla_sparse_mla_decode_interface(
            inputs["q"],
            inputs["k_cache_quantized"],
            inputs["indices_in_kvcache"],
            topk_length=inputs["topk_length"],
            sm_scale=sm_scale,
            d_v=DV,
        )
        assert torch.allclose(
            triton_out.float(),
            flashmla_out.float(),
            atol=1e-1,
            rtol=1e-1,
        ), "Triton sparse decode-compatible output does not match FlashMLA sparse decode output"
        print("Triton sparse decode-compatible output matches FlashMLA sparse decode output!")
        ref_out = flashmla_out

    if check_tle:
        tle_out_flat, _ = tle_sparse_mla_fwd_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
        )
        tle_out = tle_out_flat.view(B, S, H, DV)
        assert torch.allclose(
            tle_out.float(),
            ref_out.float(),
            atol=1e-1,
            rtol=1e-1,
        ), "TLE sparse decode-compatible output does not match reference"
        print("TLE sparse decode-compatible output matches reference!")

    if check_tilelang:
        if not _HAVE_TILELANG:
            raise RuntimeError("TileLang is not installed, cannot run TileLang sparse decode correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        tilelang_out_flat, _ = tilelang_sparse_mla_fwd_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
            block_I=resolved_block_i,
            num_stages=tilelang_num_stages,
            threads=tilelang_threads,
        )
        tilelang_out = tilelang_out_flat.view(B, S, H, DV)
        assert torch.allclose(tilelang_out.float(), ref_out.float(), atol=1e-1,
                              rtol=1e-1), ("TileLang sparse decode-compatible output does not match reference")
        print("TileLang sparse decode-compatible output matches reference!")

    if check_tilelang_pipelined:
        if not _HAVE_TILELANG:
            raise RuntimeError(
                "TileLang is not installed, cannot run pipelined TileLang sparse decode correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        tilelang_out_flat, _ = tilelang_sparse_mla_fwd_pipelined_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
            block_I=resolved_block_i,
            q_start_index=0,
            kv_stride=1,
        )
        tilelang_out = tilelang_out_flat.view(B, S, H, DV)
        assert torch.allclose(
            tilelang_out.float(), ref_out.float(), atol=1e-1,
            rtol=1e-1), ("Pipelined TileLang sparse decode-compatible output does not match reference")
        print("Pipelined TileLang sparse decode-compatible output matches reference!")

    if check_tilelang_seesaw:
        if not _HAVE_TILELANG:
            raise RuntimeError("TileLang is not installed, cannot run seesaw TileLang sparse decode correctness check")
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)
        tilelang_out_flat, _ = tilelang_sparse_mla_fwd_seesaw_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
            block_I=resolved_block_i,
            q_start_index=0,
            kv_stride=1,
        )
        tilelang_out = tilelang_out_flat.view(B, S, H, DV)
        assert torch.allclose(tilelang_out.float(), ref_out.float(), atol=1e-1,
                              rtol=1e-1), ("Seesaw TileLang sparse decode-compatible output does not match reference")
        print("Seesaw TileLang sparse decode-compatible output matches reference!")


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
    warmup=BENCH_DEFAULT_WARMUP_MS,
    rep=BENCH_DEFAULT_REP_MS,
    check_outputs=True,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
    input_mode="flashmla",
    seed=BENCH_DEFAULT_SEED,
):
    is_causal = input_mode == "causal"
    sm_scale = 0.5 if input_mode == "flashmla" else None
    q, kv, indices, topk_length = _get_bench_sparse_mla_inputs(
        B,
        S,
        SKV,
        H,
        HKV,
        DQK,
        topk,
        dtype,
        seed=seed,
        input_mode=input_mode,
    )
    results = []

    def run_triton():
        return triton_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, sm_scale=sm_scale, d_v=DV,
                                               is_causal=is_causal)

    triton_out, _ = run_triton()
    triton_ms = _bench_ms(run_triton, warmup=warmup, rep=rep)
    triton_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, triton_ms)
    results.append(("triton", triton_ms, triton_tflops))

    tle_out = None
    tilelang_out = None
    tilelang_pipelined_out = None
    tilelang_seesaw_out = None
    flashmla_out = None

    def run_tle():
        return tle_sparse_mla_fwd_interface(q, kv, indices, topk_length=topk_length, sm_scale=sm_scale, d_v=DV,
                                            is_causal=is_causal)

    try:
        tle_out, _ = run_tle()
        tle_ms = _bench_ms(run_tle, warmup=warmup, rep=rep)
        tle_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, tle_ms)
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
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
                block_I=resolved_block_i,
                num_stages=tilelang_num_stages,
                threads=tilelang_threads,
            )

        try:
            tilelang_out, _ = run_tilelang()
            tilelang_ms = _bench_ms(run_tilelang, warmup=warmup, rep=rep)
            tilelang_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, tilelang_ms)
            results.append(("tilelang", tilelang_ms, tilelang_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"TileLang bench skipped due to compile/runtime error: {exc}")
    else:
        print("TileLang is not installed, skip TileLang bench.")

    if _HAVE_TILELANG:

        def run_tilelang_pipelined():
            return tilelang_sparse_mla_fwd_pipelined_interface(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
                block_I=tilelang_block_I,
                q_start_index=0,
                kv_stride=1,
            )

        try:
            tilelang_pipelined_out, _ = run_tilelang_pipelined()
            tilelang_pipelined_ms = _bench_ms(run_tilelang_pipelined, warmup=warmup, rep=rep)
            tilelang_pipelined_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV,
                                                                            tilelang_pipelined_ms)
            results.append(("tilelang-pipelined", tilelang_pipelined_ms, tilelang_pipelined_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"Pipelined TileLang bench skipped due to compile/runtime error: {exc}")

        def run_tilelang_seesaw():
            return tilelang_sparse_mla_fwd_seesaw_interface(
                q,
                kv,
                indices,
                topk_length=topk_length,
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=is_causal,
                block_I=tilelang_block_I,
                q_start_index=0,
                kv_stride=1,
            )

        try:
            tilelang_seesaw_out, _ = run_tilelang_seesaw()
            tilelang_seesaw_ms = _bench_ms(run_tilelang_seesaw, warmup=warmup, rep=rep)
            tilelang_seesaw_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, tilelang_seesaw_ms)
            results.append(("tilelang-seesaw", tilelang_seesaw_ms, tilelang_seesaw_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"Seesaw TileLang bench skipped due to compile/runtime error: {exc}")

    if _HAVE_FLASHMLA:
        prepared_flashmla = _prepare_flashmla_sparse_prefill_args(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=sm_scale,
            d_v=DV,
        )

        def run_flashmla():
            return _run_flashmla_sparse_prefill_prepared(prepared_flashmla)

        try:
            flashmla_raw_out, flashmla_raw_lse = run_flashmla()
            flashmla_out = flashmla_raw_out.reshape(B, S, prepared_flashmla["padded_h"], DV)[:, :, :H, :]
            flashmla_ms = _bench_ms(run_flashmla, warmup=warmup, rep=rep)
            flashmla_tflops = _sparse_mla_tflops_from_topk_length(topk_length, H, DQK, DV, flashmla_ms)
            results.append(("flashmla", flashmla_ms, flashmla_tflops))
        except Exception as exc:  # pragma: no cover - depends on flashmla/runtime constraints
            print(f"FlashMLA bench skipped due to compile/runtime error: {exc}")
    else:
        print("FlashMLA is not installed, skip FlashMLA bench.")

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
        if tilelang_pipelined_out is not None:
            assert torch.allclose(
                triton_out.float(),
                tilelang_pipelined_out.float(),
                atol=1e-1,
                rtol=1e-1,
            ), "Triton output does not match pipelined TileLang output"
            print("Triton and pipelined TileLang outputs match.")
        if tilelang_seesaw_out is not None:
            assert torch.allclose(
                triton_out.float(),
                tilelang_seesaw_out.float(),
                atol=1e-1,
                rtol=1e-1,
            ), "Triton output does not match seesaw TileLang output"
            print("Triton and seesaw TileLang outputs match.")
        if flashmla_out is not None:
            assert torch.allclose(
                triton_out.float(),
                flashmla_out.float(),
                atol=1e-1,
                rtol=1e-1,
            ), "Triton output does not match FlashMLA output"
            print("Triton and FlashMLA outputs match.")


def bench_sparse_mla_decode(
    B=128,
    S=2,
    SKV=32768,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=2048,
    dtype=torch.bfloat16,
    warmup=BENCH_DEFAULT_WARMUP_MS,
    rep=BENCH_DEFAULT_REP_MS,
    check_outputs=True,
    tilelang_block_I=64,
    tilelang_num_stages=2,
    tilelang_threads=256,
    block_size=64,
):
    inputs = _build_flashmla_sparse_decode_inputs(
        B=B,
        S=S,
        SKV=SKV,
        H=H,
        HKV=HKV,
        DQK=DQK,
        topk=topk,
        dtype=dtype,
        seed=4,
        block_size=block_size,
        is_varlen=True,
    )
    sm_scale = inputs["sm_scale"]
    results = []

    def run_triton():
        return triton_sparse_mla_fwd_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
        )

    triton_out_flat, _ = run_triton()
    triton_out = triton_out_flat.view(B, S, H, DV)
    triton_ms = _bench_ms(run_triton, warmup=warmup, rep=rep)
    triton_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV, triton_ms)
    results.append(("triton", triton_ms, triton_tflops))

    tle_out = None
    tilelang_out = None
    tilelang_pipelined_out = None
    tilelang_seesaw_out = None
    flashmla_out = None

    def run_tle():
        return tle_sparse_mla_fwd_interface(
            inputs["q_flat"],
            inputs["kv_flat"],
            inputs["indices_flat"],
            topk_length=inputs["topk_length_flat"],
            sm_scale=sm_scale,
            d_v=DV,
            is_causal=False,
        )

    try:
        tle_out_flat, _ = run_tle()
        tle_out = tle_out_flat.view(B, S, H, DV)
        tle_ms = _bench_ms(run_tle, warmup=warmup, rep=rep)
        tle_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV, tle_ms)
        results.append(("tle", tle_ms, tle_tflops))
    except Exception as exc:  # pragma: no cover - depends on tle/runtime constraints
        print(f"TLE decode bench skipped due to compile/runtime error: {exc}")

    if _HAVE_TILELANG:
        resolved_block_i = _resolve_tilelang_block_i(topk, tilelang_block_I)

        def run_tilelang():
            return tilelang_sparse_mla_fwd_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
                block_I=resolved_block_i,
                num_stages=tilelang_num_stages,
                threads=tilelang_threads,
            )

        try:
            tilelang_out_flat, _ = run_tilelang()
            tilelang_out = tilelang_out_flat.view(B, S, H, DV)
            tilelang_ms = _bench_ms(run_tilelang, warmup=warmup, rep=rep)
            tilelang_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV, tilelang_ms)
            results.append(("tilelang", tilelang_ms, tilelang_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"TileLang decode bench skipped due to compile/runtime error: {exc}")

        def run_tilelang_pipelined():
            return tilelang_sparse_mla_fwd_pipelined_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
                block_I=resolved_block_i,
                q_start_index=0,
                kv_stride=1,
            )

        try:
            tilelang_pipelined_out_flat, _ = run_tilelang_pipelined()
            tilelang_pipelined_out = tilelang_pipelined_out_flat.view(B, S, H, DV)
            tilelang_pipelined_ms = _bench_ms(run_tilelang_pipelined, warmup=warmup, rep=rep)
            tilelang_pipelined_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV,
                                                                            tilelang_pipelined_ms)
            results.append(("tilelang-pipelined", tilelang_pipelined_ms, tilelang_pipelined_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"Pipelined TileLang decode bench skipped due to compile/runtime error: {exc}")

        def run_tilelang_seesaw():
            return tilelang_sparse_mla_fwd_seesaw_interface(
                inputs["q_flat"],
                inputs["kv_flat"],
                inputs["indices_flat"],
                topk_length=inputs["topk_length_flat"],
                sm_scale=sm_scale,
                d_v=DV,
                is_causal=False,
                block_I=resolved_block_i,
                q_start_index=0,
                kv_stride=1,
            )

        try:
            tilelang_seesaw_out_flat, _ = run_tilelang_seesaw()
            tilelang_seesaw_out = tilelang_seesaw_out_flat.view(B, S, H, DV)
            tilelang_seesaw_ms = _bench_ms(run_tilelang_seesaw, warmup=warmup, rep=rep)
            tilelang_seesaw_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV,
                                                                         tilelang_seesaw_ms)
            results.append(("tilelang-seesaw", tilelang_seesaw_ms, tilelang_seesaw_tflops))
        except Exception as exc:  # pragma: no cover - depends on tilelang/runtime constraints
            print(f"Seesaw TileLang decode bench skipped due to compile/runtime error: {exc}")
    else:
        print("TileLang is not installed, skip TileLang decode bench.")

    if _HAVE_FLASHMLA:
        tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()

        def run_flashmla():
            return flashmla_sparse_mla_decode_interface(
                inputs["q"],
                inputs["k_cache_quantized"],
                inputs["indices_in_kvcache"],
                topk_length=inputs["topk_length"],
                sm_scale=sm_scale,
                d_v=DV,
                tile_scheduler_metadata=tile_scheduler_metadata,
            )

        try:
            flashmla_out, _ = run_flashmla()
            flashmla_ms = _bench_ms(run_flashmla, warmup=warmup, rep=rep)
            flashmla_tflops = _sparse_mla_tflops_from_topk_length(inputs["topk_length_flat"], H, DQK, DV, flashmla_ms)
            results.append(("flashmla", flashmla_ms, flashmla_tflops))
        except Exception as exc:  # pragma: no cover - depends on flashmla/runtime constraints
            print(f"FlashMLA decode bench skipped due to compile/runtime error: {exc}")
    else:
        print("FlashMLA is not installed, skip FlashMLA decode bench.")

    print(f"{'provider':<18}{'ms':>10}{'tflops':>12}{'speedup':>12}")
    for name, ms, tflops in results:
        print(f"{name:<18}{ms:>10.3f}{tflops:>12.2f}{(triton_ms / ms):>12.2f}x")

    if check_outputs:
        if flashmla_out is not None:
            assert torch.allclose(triton_out.float(), flashmla_out.float(), atol=1e-1,
                                  rtol=1e-1), ("Triton decode-compatible output does not match FlashMLA decode output")
            print("Triton and FlashMLA decode outputs match.")
        if tle_out is not None:
            assert torch.allclose(triton_out.float(), tle_out.float(), atol=1e-1,
                                  rtol=1e-1), ("Triton decode-compatible output does not match TLE output")
            print("Triton and TLE decode-compatible outputs match.")
        if tilelang_out is not None:
            assert torch.allclose(triton_out.float(), tilelang_out.float(), atol=1e-1,
                                  rtol=1e-1), ("Triton decode-compatible output does not match TileLang output")
            print("Triton and TileLang decode-compatible outputs match.")
        if tilelang_pipelined_out is not None:
            assert torch.allclose(
                triton_out.float(), tilelang_pipelined_out.float(), atol=1e-1,
                rtol=1e-1), ("Triton decode-compatible output does not match pipelined TileLang output")
            print("Triton and pipelined TileLang decode-compatible outputs match.")
        if tilelang_seesaw_out is not None:
            assert torch.allclose(triton_out.float(), tilelang_seesaw_out.float(), atol=1e-1,
                                  rtol=1e-1), ("Triton decode-compatible output does not match seesaw TileLang output")
            print("Triton and seesaw TileLang decode-compatible outputs match.")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["test", "test-decode", "bench", "bench-single", "bench-decode",
                                 "bench-decode-single"], default="bench")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--SKV", type=int, default=4096)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--HKV", type=int, default=1)
    parser.add_argument("--DQK", type=int, default=576)
    parser.add_argument("--DV", type=int, default=512)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=BENCH_DEFAULT_WARMUP_MS,
                        help="Benchmark warmup budget in milliseconds.")
    parser.add_argument("--rep", type=int, default=BENCH_DEFAULT_REP_MS,
                        help="Benchmark measurement budget in milliseconds.")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--bench-input-mode", choices=["flashmla", "causal"], default="flashmla")
    parser.add_argument("--seed", type=int, default=BENCH_DEFAULT_SEED, help="Seed for benchmark input generation.")
    parser.add_argument("--skip-output-check", action="store_true")
    parser.add_argument("--skip-tle-check", action="store_true")
    parser.add_argument("--check-tilelang", action="store_true")
    parser.add_argument("--check-tilelang-pipelined", action="store_true")
    parser.add_argument("--check-tilelang-seesaw", action="store_true")
    parser.add_argument("--check-flashmla", action="store_true")
    parser.add_argument("--tilelang-block-I", type=int, default=64)
    parser.add_argument("--tilelang-num-stages", type=int, default=2)
    parser.add_argument("--tilelang-threads", type=int, default=256)
    parser.add_argument("--decode-block-size", type=int, default=64)
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
            check_tilelang_pipelined=args.check_tilelang_pipelined,
            check_tilelang_seesaw=args.check_tilelang_seesaw,
            check_flashmla=args.check_flashmla,
            tilelang_block_I=args.tilelang_block_I,
            tilelang_num_stages=args.tilelang_num_stages,
            tilelang_threads=args.tilelang_threads,
        )
    elif args.mode == "test-decode":
        test_sparse_mla_decode(
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
            check_tilelang_pipelined=args.check_tilelang_pipelined,
            check_tilelang_seesaw=args.check_tilelang_seesaw,
            check_flashmla=args.check_flashmla,
            tilelang_block_I=args.decode_block_size,
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
            input_mode=args.bench_input_mode,
            seed=args.seed,
        )
    elif args.mode == "bench-decode-single":
        bench_sparse_mla_decode(
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
            block_size=args.decode_block_size,
        )
    elif args.mode == "bench-decode":
        run_decode_bench_table(
            warmup=args.warmup,
            rep=args.rep,
            show_plots=args.show_plots,
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
            input_mode=args.bench_input_mode,
            seed=args.seed,
        )
