import torch
import torch_musa
import triton
import triton.language as tl

atol = 3e-2


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


def test_fwd_grouped_kernel_stage1_rdn():
    torch.manual_seed(42)  # 种子不变，随机数不变
    batch_size = 1
    head_num = 16
    kv_group_num = 16
    Lk = 576
    Lv = 512
    NUM_KV_SPLITS = 4
    page_size = 16
    q = torch.randn((batch_size, head_num, Lk), dtype=torch.bfloat16, device="musa:0")
    k_buffer = torch.randn((64, 16, 1, Lk), dtype=torch.bfloat16, device="musa:0")
    v_buffer = torch.randn((64, 16, 1, Lv), dtype=torch.bfloat16, device="musa:0")
    att_out_mma = torch.randn(
        (batch_size, head_num, NUM_KV_SPLITS, Lv + 1),
        dtype=torch.float32,
        device="musa:0",
    )
    att_out_fma = att_out_mma.clone()
    Req_to_tokens = torch.randint(0, 64, (batch_size, 1), dtype=torch.int32, device="musa:0")
    B_Seqlen = torch.tensor([9], dtype=torch.int32, device="musa:0")
    BLOCK_DMODEL = 512
    BLOCK_DPE = 64
    BLOCK_DV = 512
    BLOCK = 32
    BLOCK_H = 16
    logit_cap = 0.0

    extra_kargs = {}
    grid = (
        batch_size,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    # use mma
    _fwd_grouped_kernel_stage1[grid](
        q, k_buffer, v_buffer, 0.1147213867929261, Req_to_tokens, B_Seqlen, att_out_mma, Req_to_tokens.stride(0),
        q.stride(0), q.stride(1), k_buffer.stride(-3), k_buffer.stride(-2), v_buffer.stride(-3), v_buffer.stride(-2),
        att_out_mma.stride(0), att_out_mma.stride(1), att_out_mma.stride(2), kv_group_num=kv_group_num,
        q_head_num=head_num, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV, BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS, PAGE_SIZE=page_size, logit_cap=logit_cap, num_warps=1,
        num_stages=2, Lk=Lk, Lv=Lv, **extra_kargs, en_wmma=True,  # use mma
    )

    # use fma
    _fwd_grouped_kernel_stage1[grid](
        q, k_buffer, v_buffer, 0.1147213867929261, Req_to_tokens, B_Seqlen, att_out_fma, Req_to_tokens.stride(0),
        q.stride(0), q.stride(1), k_buffer.stride(-3), k_buffer.stride(-2), v_buffer.stride(-3), v_buffer.stride(-2),
        att_out_fma.stride(0), att_out_fma.stride(1), att_out_fma.stride(2), kv_group_num=kv_group_num,
        q_head_num=head_num, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE, BLOCK_DV=BLOCK_DV, BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS, PAGE_SIZE=page_size, logit_cap=logit_cap, num_warps=4,
        num_stages=2, Lk=Lk, Lv=Lv, **extra_kargs, en_wmma=False,  # use fma
    )

    correct = torch.allclose(att_out_mma, att_out_fma, atol=atol)
    # print(f"att_out_fma: {att_out_fma}")
    # print(f"att_out_mma: {att_out_mma}")
    # print(f"fwd_grouped_kernel_stage1 test: {correct}")
    assert correct, "fwd_grouped_kernel_stage1 test failed: outputs do not match!"


if __name__ == "__main__":
    test_fwd_grouped_kernel_stage1_rdn()
