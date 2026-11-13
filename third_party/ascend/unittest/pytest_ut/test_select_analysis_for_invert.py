# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def cal_atten_mask_kernel(
    QK_ptr,
    Indices_ptr,
    stride_qk_m,
    stride_qk_n,
    stride_ik,
    SEQ_LEN: tl.constexpr,
    sparse_block_size: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    TOPK_BASE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    idx_sub_sbs = pid_n
    cur_s1 = pid_m * BLOCK_SBS
    cur_s2 = cur_s1 + BLOCK_SBS

    if cur_s1 >= SEQ_LEN:
        return

    beg_sbs = idx_sub_sbs * BLOCK_SBS // sparse_block_size
    end_sbs = ((idx_sub_sbs + 1) * BLOCK_SBS) // sparse_block_size

    valid_col_end = cur_s1 + (cur_s2 - cur_s1)

    offs_m = cur_s1 + tl.arange(0, BLOCK_SBS)
    offs_n_base = idx_sub_sbs * BLOCK_SBS
    offs_n = offs_n_base + tl.arange(0, BLOCK_SBS)

    mask_m = offs_m < SEQ_LEN
    mask_n = offs_n < SEQ_LEN
    mask_load = mask_m[:, None] & mask_n[None, :]

    qk_ub = tl.load(QK_ptr + offs_m[:, None] * stride_qk_m + offs_n[None, :] * stride_qk_n, mask=mask_load, other=0.0)

    for idx_k in range(beg_sbs, end_sbs):
        idx_s2 = tl.load(Indices_ptr + TOPK_BASE + idx_k * stride_ik)
        if idx_s2 != -1 and idx_s2 * sparse_block_size > valid_col_end:
            idx_lower_sbs = idx_k * sparse_block_size - \
                idx_sub_sbs * BLOCK_SBS
            idx_higher_sbs = (idx_k + 1) * sparse_block_size - \
                idx_sub_sbs * BLOCK_SBS
            mask_lower_sbs = tl.arange(0, BLOCK_SBS) >= idx_lower_sbs
            mask_higher_sbs = tl.arange(0, BLOCK_SBS) < idx_higher_sbs
            qk_ub = tl.where((mask_lower_sbs & mask_higher_sbs)[None, :], float("-inf"), qk_ub)

    tl.store(QK_ptr + offs_m[:, None] * stride_qk_m + offs_n[None, :] * stride_qk_n, qk_ub, mask=mask_load)


def launch_cal_atten_mask(qk_tensor, indices_tensor, sparse_block_size=64, block_sbs=128):
    """
    qk_tensor: (SEQ_LEN, SEQ_LEN)
    indices_tensor: (K,) / (BATCH, K, ...)
    """
    assert qk_tensor.is_contiguous()
    M, N = qk_tensor.shape

    stride_qk_m = qk_tensor.stride(0)
    stride_qk_n = qk_tensor.stride(1)

    stride_ik = 1
    topk_base = 0

    grid = (triton.cdiv(M, block_sbs), triton.cdiv(N, block_sbs))
    cal_atten_mask_kernel[grid](
        qk_tensor,
        indices_tensor,
        stride_qk_m,
        stride_qk_n,
        stride_ik,
        SEQ_LEN=M,
        sparse_block_size=sparse_block_size,
        BLOCK_SBS=block_sbs,
        TOPK_BASE=topk_base,
    )
    return qk_tensor


def torch_cal_atten_mask(
    qk,
    indices,
    sparse_block_size,
    block_sbs,
    topk_base=0,
):
    device = qk.device
    dtype = qk.dtype
    M, N = qk.shape

    row_ids = torch.arange(M, device=device).unsqueeze(1)
    col_ids = torch.arange(N, device=device).unsqueeze(0)

    k_idx_global = col_ids // sparse_block_size
    lookup_idx = k_idx_global + topk_base
    max_valid_idx = indices.numel() - 1

    valid_lookup = (lookup_idx >= 0) & (lookup_idx <= max_valid_idx)
    safe_lookup_idx = lookup_idx.clamp(0, max_valid_idx)
    idx_s2_map = indices.gather(0, safe_lookup_idx.squeeze(0)).unsqueeze(0)
    idx_s2_map = torch.where(valid_lookup, idx_s2_map, torch.tensor(-1, device=device))

    row_block_ends = ((row_ids // block_sbs) + 1) * block_sbs
    row_block_ends = torch.min(row_block_ends, torch.tensor(N, device=device))

    start_pos_k_map = idx_s2_map * sparse_block_size
    cond_valid = (idx_s2_map != -1)
    cond_exceed = (start_pos_k_map > row_block_ends)
    final_mask = cond_valid & cond_exceed

    qk_out = torch.where(final_mask, torch.tensor(float("-inf"), dtype=dtype, device=device), qk)
    return qk_out


@pytest.mark.parametrize('param_list', [['float32', 1024, 128, 64]])
def test_divsiop_select_analysis1(param_list):
    dtype, SEQ_LEN, BLOCK_SBS, SPARSE_BLOCK = param_list
    qk = torch.zeros((SEQ_LEN, SEQ_LEN), dtype=eval('torch.' + dtype), device='npu')
    K_SIZE = 20
    indices = torch.full((K_SIZE, ), -1, dtype=torch.int32, device='npu')
    indices[10] = 20
    qk_ref = torch_cal_atten_mask(qk.clone(), indices, sparse_block_size=SPARSE_BLOCK, block_sbs=BLOCK_SBS)
    qk_cal = launch_cal_atten_mask(qk, indices, sparse_block_size=SPARSE_BLOCK, block_sbs=BLOCK_SBS)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)
