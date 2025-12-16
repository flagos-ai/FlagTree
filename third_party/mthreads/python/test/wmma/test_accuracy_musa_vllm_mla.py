import logging
import os

import triton
import triton.language as tl

import pytest
import torch
import torch_musa

# from vllm.platforms import current_platform

# is_hip_ = current_platform.is_rocm()
is_hip_ = False

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance
# and accuracy.
logger.warning("The following error message 'operation scheduled before its operands' "
               "can be ignored.")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(996)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :])
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv)

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + Lv)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 64
    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    num_warps = 4 if kv_group_num == 1 else 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :])
        qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None])
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None])
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob + cur_head[:, None] * stride_mid_oh + split_kv_id * stride_mid_os +
                      offs_dv[None, :])

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + Lv)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
    en_wmma,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if is_hip_ and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    # BLOCK_H = 16
    BLOCK_H = 32
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
        en_wmma=en_wmma,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(logits, q, o, v_buffer, b_seq_len, num_kv_splits, en_wmma):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
        en_wmma=en_wmma,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits)


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
    en_wmma=False,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        req_to_token,
        b_seq_len,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        en_wmma,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len, num_kv_splits, en_wmma)


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
    en_wmma=False,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[-2]

    assert kv_group_num != 1  # force to use decode_attention_fwd_grouped

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            req_to_token,
            b_seq_len,
            attn_logits,
            num_kv_splits,
            sm_scale,
            page_size,
            logit_cap,
            en_wmma=en_wmma,
        )


def cdiv(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("B", [3, 5])
@pytest.mark.parametrize("L", [1027, 1025])
@pytest.mark.parametrize("H_Q", [32])
# @pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("H_KV", [8])
@pytest.mark.parametrize("D_QK", [128, 192, 576])
@pytest.mark.parametrize("D_V", [128, 512])
@pytest.mark.parametrize("CACHE_SIZE", [16384])
@pytest.mark.parametrize("PAGE_SIZE", [1, 16])
def test_decode_attention(B, L, H_Q, H_KV, D_QK, D_V, CACHE_SIZE, PAGE_SIZE):
    assert CACHE_SIZE % PAGE_SIZE == 0
    dtype = torch.bfloat16
    seq_len = L  # This represents the number of tokens already in the sequence
    sm_scale = 1.0 / (D_QK**0.5)
    num_kv_splits = 8

    num_pages_per_batch = cdiv(seq_len, PAGE_SIZE)
    req_to_page = torch.randint(0, CACHE_SIZE // PAGE_SIZE, (B, num_pages_per_batch, 1), device="musa")
    req_to_token = req_to_page * PAGE_SIZE
    req_to_token = req_to_token.expand(B, num_pages_per_batch, PAGE_SIZE)
    req_to_token = req_to_token + torch.arange(PAGE_SIZE, device="musa").view(1, 1, -1)
    req_to_token = req_to_token.view(B, -1)
    req_to_token = req_to_token[:, :seq_len].contiguous()

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D_QK, dtype=dtype, device="musa")

    # k_buffer and v_buffer represent all previous tokens
    # Page size is 1.
    k_buffer = torch.randn(CACHE_SIZE, H_KV, D_QK, dtype=dtype, device="musa")
    v_buffer = torch.randn(CACHE_SIZE, H_KV, D_V, dtype=dtype, device="musa")

    # # o will have the same shape as q
    # o = torch.zeros(B, H_Q, D_V, dtype=dtype, device="musa")

    # b_seq_len = torch.full((B, ), seq_len, device="musa")

    # attn_logits = torch.empty(
    #     (B, H_Q, num_kv_splits, D_V + 1),
    #     dtype=torch.float32,
    #     device="musa",
    # )

    # # Call the original implementation.
    # decode_attention_fwd(
    #     q,
    #     k_buffer,
    #     v_buffer,
    #     o,
    #     req_to_token,
    #     b_seq_len,
    #     attn_logits,
    #     num_kv_splits,
    #     sm_scale,
    # )

    # # Page size can be larger than 1.
    # k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    # v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    # o1 = torch.zeros_like(o)

    # decode_attention_fwd(
    #     q,
    #     k_buffer,
    #     v_buffer,
    #     o1,
    #     req_to_page,
    #     b_seq_len,
    #     attn_logits,
    #     num_kv_splits,
    #     sm_scale,
    #     PAGE_SIZE,
    # )

    # o_save_path = f"./B_{B}_L_{L}_H_Q_{H_Q}_H_KV_{H_KV}_D_QK_{D_QK}_D_V_{D_V}_CACHE_SIZE_{CACHE_SIZE}_PAGE_SIZE_{PAGE_SIZE}_fma_o.cpu"
    # o1_save_path = f"./B_{B}_L_{L}_H_Q_{H_Q}_H_KV_{H_KV}_D_QK_{D_QK}_D_V_{D_V}_CACHE_SIZE_{CACHE_SIZE}_PAGE_SIZE_{PAGE_SIZE}_fma_o1.cpu"
    # if not os.getenv("ENABLE_MUSA_MMA"):
    #     print("========FMA MODE========")
    #     if not (os.path.exists(o_save_path) and os.path.exists(o1_save_path)):
    #         print("FMA saving o.cpu")
    #         torch.save(o.cpu(), o_save_path)
    #         print("FMA saving o1.cpu")
    #         torch.save(o1.cpu(), o1_save_path)
    #     else:
    #         print("FMA loading o.cpu")
    #         REF_O_TENSOR = torch.load(o_save_path).musa()
    #         correct0 = torch.allclose(REF_O_TENSOR, o)
    #         print("FMA loading o1.cpu")
    #         REF_O1_TENSOR = torch.load(o1_save_path).musa()
    #         correct1 = torch.allclose(REF_O1_TENSOR, o1)
    #         print(f"fma decode_attention_fwd test o is pass: {correct0}")
    #         print(f"fma decode_attention_fwd test o1 is pass: {correct1}")
    #         assert (correct0 and correct1)
    # else:
    #     print("========MMA MODE========")
    #     print("MMA loading o.cpu")
    #     REF_O_TENSOR = torch.load(o_save_path).musa()
    #     correct0 = torch.allclose(REF_O_TENSOR, o, atol=1e-2)
    #     print("MMA loading o1.cpu")
    #     REF_O1_TENSOR = torch.load(o1_save_path).musa()
    #     correct1 = torch.allclose(REF_O1_TENSOR, o1, atol=1e-2)
    #     print(f"mma decode_attention_fwd test o is pass: {correct0}")
    #     print(f"mma decode_attention_fwd test o1 is pass: {correct1}")
    #     assert (correct0 and correct1)

    # assert torch.allclose(o, o1)

    b_seq_len = torch.full((B, ), seq_len, device="musa")

    attn_logits = torch.empty(
        (B, H_Q, num_kv_splits, D_V + 1),
        dtype=torch.float32,
        device="musa",
    )

    o_fma = torch.zeros(B, H_Q, D_V, dtype=dtype, device="musa")
    o_wmma = torch.zeros(B, H_Q, D_V, dtype=dtype, device="musa")
    o1_fma = torch.zeros_like(o_fma)
    o1_wmma = torch.zeros_like(o_wmma)

    # Call the original implementation.
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o_fma,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        en_wmma=False,
    )
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o_wmma,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        en_wmma=True,
    )

    # Page size can be larger than 1.
    k_buffer = k_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_QK)
    v_buffer = v_buffer.view(CACHE_SIZE // PAGE_SIZE, PAGE_SIZE, H_KV, D_V)

    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1_fma,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
        en_wmma=False,
    )
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o1_wmma,
        req_to_page,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        PAGE_SIZE,
        en_wmma=True,
    )

    correct0 = torch.allclose(o_fma, o_wmma, atol=1e-2)
    correct1 = torch.allclose(o1_fma, o1_wmma, atol=1e-2)
    # print(f"mma decode_attention_fwd test o is pass: {correct0}")
    # print(f"mma decode_attention_fwd test o1 is pass: {correct1}")
    assert correct0 and correct1
