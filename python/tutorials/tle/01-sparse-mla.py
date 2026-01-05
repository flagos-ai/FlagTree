# flagtree
"""
Sparse MLA Forward
==================

This module implements a Triton kernel for the forward pass of a sparse MLA (Multi-Headed Attention) mechanism.
It demonstrates the use of Triton's TLE (Triton Language Extensions) for efficient memory access and computation.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

spar_mla_fwd_configs = [
    triton.Config({'num_stages': 4, 'num_warps': 8}),
    # triton.Config({'num_stages': 2, 'num_warps': 4}),
]


@triton.autotune(  # Decorate the kernel
    configs=spar_mla_fwd_configs,
    key=['K', 'is_causal'],
)
@triton.jit
def triton_sparse_mla_fwd(
    q,
    kv,
    indices,
    sm_scale: tl.constexpr,
    output,
    lse,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kvb, stride_kvg, stride_kvn, stride_kvd,
    stride_tb, stride_tg, stride_tm, stride_tt,  # topk，for indices
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    B: tl.constexpr,
    SQ: tl.constexpr,  # seqlen
    SKV: tl.constexpr,
    K: tl.constexpr,  # topk
    D: tl.constexpr,  # QKV dim
    TD: tl.constexpr,  # tail dim
    DP: tl.constexpr,
    TDP: tl.constexpr,
    H: tl.constexpr,  # q_head_dim
    G: tl.constexpr,  # group_size
    VG: tl.constexpr,  # H/G KV groups
    BK: tl.constexpr,
    BH: tl.constexpr,
    is_causal: tl.constexpr
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

    max_prev = tl.full([BH], float('-inf'), dtype=tl.float32)
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
            kv_blk = tle.load(kv_ptr, kv_msk, other=0.0, is_async=True)  # [DP, BK]

            tkv_ptr = tkv_base + offs_td[:, None] * stride_kvd + kv_ids[None, :] * stride_kvn
            tkv_msk = mask_td[:, None] & mask_ids[None, :]
            tkv_blk = tle.load(tkv_ptr, tkv_msk, other=0.0, is_async=False)  # [TDP, BK]

            qk = tl.dot(tq_blk, tkv_blk, out_dtype=tl.float32)
            qk = tl.dot(q_blk, kv_blk, qk, out_dtype=tl.float32)

            qk = tl.where(mask_ids[None, :], qk, float('-inf'))  # [BH, BK]

            new_max = tl.maximum(max_prev, tl.max(qk, axis=1))
            alpha = tl.math.exp2((max_prev - new_max) * log_scale)
            exp_qk = tl.math.exp2(qk * log_scale - new_max[:, None] * log_scale)
            sum_qk = tl.sum(exp_qk, axis=1)
            sum_exp = sum_exp * alpha + sum_qk
            acc = acc * alpha[:, None]
            exp_qk = exp_qk.to(tl.bfloat16)
            acc = tl.dot(exp_qk, tl.trans(kv_blk), acc, out_dtype=tl.float32)  # [BH, BK] @ [BK, DP] = [BH, DP]

            max_prev = new_max

    out_vals = acc / sum_exp[:, None]
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_od[None, :] * stride_od
    o_msk = mask_h[:, None] & mask_od[None, :]
    tl.store(o_ptr, out_vals.to(q_blk.dtype), o_msk)

    fin_log = max_prev * log_scale + tl.math.log2(sum_exp.to(tl.float32))  # lse / ln2
    l_ptr = l_base + offs_h * stride_lh
    l_msk = mask_h
    tl.store(l_ptr, fin_log.to(q_blk.dtype), l_msk)


def triton_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, return_p_sum: bool = False, d_v=512):
    is_causal = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    B, SQ, H, DT = q.shape
    _, S, VG, _ = kv.shape

    # assert DT == 576, "you should assign dim otherwise"
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
        sm_scale = DT ** -0.5
    BH = 32
    NH = triton.cdiv(G, BH)
    BK = 32
    output = torch.zeros((B, SQ, H, D), device=q.device, dtype=q.dtype)
    lse = torch.full((B, SQ, H), float('-inf'), device=q.device, dtype=q.dtype)
    grid = (B, SQ, VG * NH)  # (SQ//BQ, B*H)
    triton_sparse_mla_fwd[grid](
        q, kv, indices, sm_scale,
        output, lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),  # [B, H, SQ, DT]
        kv.stride(0), kv.stride(2), kv.stride(1), kv.stride(3),  # [B, VG, SKV, DT]
        indices.stride(0), indices.stride(2), indices.stride(1), indices.stride(3),  # [B, VG, SQ, K]
        output.stride(0), output.stride(2), output.stride(1), output.stride(3),  # [B, H, SQ, D]
        lse.stride(0), lse.stride(2), lse.stride(1),  # [B, H, SQ]
        B, SQ, S, K, D, TD, DP, TDP, H, G, VG,
        BK,
        BH,
        # BD,
        is_causal
    )
    # sparse_mla_fwd[grid](q, kv, indices, output)
    return output, lse


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True, d_v=512):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    dim = d_v
    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    # dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda").view(1, -1)

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


def test_sparse_mla_fwd(B=1,
                        S=4096,
                        SKV=4096,
                        H=128,
                        HKV=1,
                        DQK=576,
                        DV=512,
                        topk=2048,
                        dtype=torch.bfloat16):
    torch.random.manual_seed(0)
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda").requires_grad_(True)
    kv = torch.randn((B, SKV, HKV, DQK), dtype=dtype, device="cuda").requires_grad_(True)

    indices = torch.full((B, S, HKV, topk), SKV, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[b, t, h, :len(i_i)] = i_i

    ref_bf16_out = ref_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)

    triton_bf16_out, triton_bf16_lse = triton_sparse_mla_fwd_interface(q, kv, indices, d_v=DV)
    print("triton bf16 done \n triton lse tensor: \n", triton_bf16_lse)
    print()

    assert torch.allclose(
        triton_bf16_out.float(),
        ref_bf16_out.float(),
        atol=1e-1,
        rtol=1e-1,
    ), "Triton sparse MLA fwd bf16 does not match reference"
    print("Triton sparse MLA fwd bf16 matches reference!")


if __name__ == "__main__":
    test_sparse_mla_fwd(
        B=1, S=128, SKV=1024, H=32, HKV=1, DQK=256 + 32, DV=256, topk=64, dtype=torch.bfloat16)
