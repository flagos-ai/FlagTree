import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common


# source: python\sglang\srt\layers\attention\triton_ops\double_sparsity_attention.py
@triton.jit
def _sparse_fwd_kernel_flash_decode_stage2(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,  # shape: [B, S]
    Topk_token_indices,  # shape: [H, B, k]
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Heavy_token_num,  # NOTE: This can be used as constexpr but we may support dynamic heavy token number in the future
    stride_req_to_tokens_b,
    stride_topk_token_indices_h,
    stride_topk_token_indices_b,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_o_eb,
    stride_mid_o_eh,
    gqa_group_size,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(Heavy_token_num, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = tl.arange(0, BLOCK_N)

    q = tl.load(Q + off_q)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(cur_batch_start_index, cur_batch_end_index, BLOCK_N):
        offs_n_new = start_n + offs_n
        topk_token_indices = tl.load(
            Topk_token_indices
            + stride_topk_token_indices_h * cur_head
            + stride_topk_token_indices_b * cur_batch
            + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch + topk_token_indices,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        )
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(
            K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        v = tl.load(
            V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0
        )

        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    need_store = 1
    for _ in range(0, need_store, 1):
        off_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + seq_start_block * stride_mid_os
            + offs_d
        )
        off_mid_o_logexpsum = (
            cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        )
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


def test__sparse_fwd_kernel_flash_decode_stage2(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    _sparse_fwd_kernel_flash_decode_stage2[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")