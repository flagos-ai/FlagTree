# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone LoRA (punica) Triton kernels extracted from vLLM.
This module intentionally avoids any vLLM imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import triton
import triton.language as tl

_LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}
_LORA_B_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}


def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    """
    Cache lora_a pointers/strides for grouped GEMM.
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_a_weights)

    if values := _LORA_A_PTR_DICT.get(key):
        return values

    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        if lora_a_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_a_weight.size(1) == 1
            lora_a_weight = lora_a_weight.squeeze(dim=1)
        else:
            assert lora_a_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        lora_ptr_tensor = lora_a_weights[0]

    if (len(set(lora_strides_d0)) > 1 or len(set(lora_strides_d1)) > 1 or len(set(lora_strides_d2)) > 1):
        raise ValueError("All LoRA weights must have the same stride.")

    _LORA_A_PTR_DICT[key] = (
        lora_ptr_tensor,
        lora_strides_d0[0],
        lora_strides_d1[0],
        lora_strides_d2[0],
    )
    return _LORA_A_PTR_DICT.get(key)


def _get_lora_b_ptr(lora_weights: list[torch.Tensor], offset_start: int, device: torch.device):
    """
    Cache lora_b pointers/strides for grouped GEMM.
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)
    if values := _LORA_B_PTR_DICT.get(key):
        return values
    slice_offset_lst = []
    tensor_ptrs = []
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    hidden_sizes = []
    slice_offset = offset_start
    for lora_b_weight in lora_weights:
        if lora_b_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_b_weight.size(1) == 1
            lora_b_weight = lora_b_weight.squeeze(dim=1)
        else:
            assert lora_b_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_b_weight.is_contiguous()
        tensor_ptrs.append(lora_b_weight.data_ptr())
        lora_strides_d0.append(lora_b_weight.stride(0))
        lora_strides_d1.append(lora_b_weight.stride(1))
        lora_strides_d2.append(lora_b_weight.stride(2))
        slice_offset_lst.append(slice_offset)
        slice_offset += lora_b_weight.size(1)
        hidden_sizes.append(lora_b_weight.size(1))

    if len(lora_weights) > 1:
        # note these are device tensors
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
        slice_start_tensor = torch.tensor(slice_offset_lst, device=device, dtype=torch.uint64)
    else:
        slice_start_tensor = slice_offset_lst[0]
        lora_ptr_tensor = lora_weights[0]

    # If each lora has the same stride, there's no need to use a
    # tensor for storage.
    if (len(set(lora_strides_d0)) == 1 and len(set(lora_strides_d1)) == 1 and len(set(lora_strides_d2)) == 1) and len(
            set(hidden_sizes)) == 1:
        lora_strides_d0_tensor = lora_strides_d0[0]
        lora_strides_d1_tensor = lora_strides_d1[0]
        lora_strides_d2_tensor = lora_strides_d2[0]
        hidden_sizes_tensor = hidden_sizes[0]
        same_stride = True

    else:
        lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
        lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
        lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)
        hidden_sizes_tensor = torch.tensor(hidden_sizes, device=device)
        same_stride = False
    # MAX_N is the maximum hidden size among all the lora_b weights
    MAX_N = max(hidden_sizes)
    _LORA_B_PTR_DICT[key] = (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor, lora_strides_d1_tensor,
                             lora_strides_d2_tensor, hidden_sizes_tensor, same_stride, MAX_N)
    return _LORA_B_PTR_DICT.get(key)


@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor
    no_lora_flag_cpu: torch.Tensor

    @staticmethod
    def make(max_loras: int, max_num_tokens: int, device: Union[torch.device, str]) -> "LoRAKernelMeta":
        token_lora_mapping = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        token_indices_sorted_by_lora_ids = torch.empty(max_num_tokens, dtype=torch.int32, device=device)
        active_lora_ids = torch.empty(max_loras + 1, dtype=torch.int32, device=device)
        num_tokens_per_lora = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
        lora_token_start_loc = torch.zeros(max_loras + 2, dtype=torch.int32, device=device)
        no_lora_flag_cpu = torch.tensor([False], dtype=torch.bool, device='cpu')
        return LoRAKernelMeta(token_lora_mapping=token_lora_mapping,
                              token_indices_sorted_by_lora_ids=token_indices_sorted_by_lora_ids,
                              active_lora_ids=active_lora_ids, num_tokens_per_lora=num_tokens_per_lora,
                              lora_token_start_loc=lora_token_start_loc, no_lora_flag_cpu=no_lora_flag_cpu)

    def _reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)
        self.no_lora_flag_cpu.fill_(False)

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        self._reset()
        no_lora = torch.all(token_lora_mapping == -1)
        self.no_lora_flag_cpu[0] = no_lora
        if no_lora:
            return

        num_tokens = token_lora_mapping.size(0)
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping, non_blocking=True)
        _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping, stable=True)
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(token_indices_sorted_by_lora_ids, non_blocking=True)
        lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
        self.active_lora_ids[:lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
        self.num_tokens_per_lora[:num_tokens_per_lora.size(0)].copy_(num_tokens_per_lora, non_blocking=True)
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1:1 + lora_token_start_loc.size(0)].copy_(lora_token_start_loc, non_blocking=True)

    def meta_args(
            self, token_nums: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.token_lora_mapping[:token_nums],
            self.token_indices_sorted_by_lora_ids[:token_nums],
            self.num_tokens_per_lora,
            self.lora_token_start_loc,
            self.active_lora_ids,
            self.no_lora_flag_cpu,
        )


@triton.jit
def mm_k(a_ptr, b_ptr, ak_stride, bn_stride, bk_stride, offset_k, K: tl.constexpr, BLOCK_M: tl.constexpr,
         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
         CAST_TYPE: tl.constexpr, b_dtype: tl.constexpr, USE_STRIDE_LOAD: tl.constexpr):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            if USE_STRIDE_LOAD:
                tiled_b = tl.load(b_ptr, stride=bn_stride)
            else:
                tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr, mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K), other=0)
            if USE_STRIDE_LOAD:
                tiled_b = tl.load(b_ptr, stride=bn_stride, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K),
                                  other=0)
            else:
                tiled_b = tl.load(b_ptr, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K), other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(b_dtype)
        accumulator += tl.dot(tiled_a, tiled_b)
        a_ptr += BLOCK_K * SPLIT_K * ak_stride
        b_ptr += BLOCK_K * SPLIT_K * bk_stride
    return accumulator


@triton.jit
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_STRIDE_LOAD: tl.constexpr,
):
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = (cur_input_ptr + ram[:, None] * input_d1_stride + offset_k[None, :] * input_d2_stride)
    b_ptr = (cur_lora_ptr + cur_lora_d0_stride * lora_index + offset_k[:, None] * cur_lora_d2_stride +
             rbn[None, :] * cur_lora_d1_stride)

    SPLIT_K = 1
    accumulator = mm_k(a_ptr, b_ptr, input_d2_stride, cur_lora_d1_stride, cur_lora_d2_stride, offset_k, K, BLOCK_M,
                       BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, CAST_TYPE, cur_lora_ptr.dtype.element_ty, USE_STRIDE_LOAD)

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    if SLICE_NUM == 1:
        cur_slice_start = slice_start_loc
    else:
        cur_slice_start = tl.load(slice_start_loc + slice_id)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = (out_ptr + ram[:, None] * output_d0_stride + offset_cn[None, :] * output_d1_stride)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))

    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
    ram,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_STRIDE_LOAD: tl.constexpr,
):
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)

    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = (input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride)
    b_ptr = (cur_lora_ptr + lora_d0_stride * lora_index + rbn[None, :] * lora_d1_stride +
             offset_k[:, None] * lora_d2_stride)

    accumulator = mm_k(a_ptr, b_ptr, input_d1_stride, lora_d1_stride, lora_d2_stride, offset_k, K, BLOCK_M, BLOCK_N,
                       BLOCK_K, EVEN_K, SPLIT_K, False, cur_lora_ptr.dtype.element_ty, USE_STRIDE_LOAD)

    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = (out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride)
    c_ptr = (cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[None, :] * output_d2_stride)
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)

    accumulator *= scaling
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)


@triton.jit
def _lora_expand_kernel(input_ptr, lora_ptr, out_ptr, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                        lora_token_start_loc, lora_ids, slice_start_loc, input_d0_stride, input_d1_stride,
                        input_d2_stride, ls_d0_ptr, ls_d1_ptr, ls_d2_ptr, output_d0_stride, output_d1_stride,
                        output_hs_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        EVEN_K: tl.constexpr, ADD_INPUTS: tl.constexpr, CAST_TYPE: tl.constexpr,
                        SLICE_NUM: tl.constexpr, SAME_STRIDE: tl.constexpr, USE_STRIDE_LOAD: tl.constexpr):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    if pid_n * BLOCK_N >= curr_N:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset)

    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_expand_kernel(pid_n, lora_id, slice_id, input_ptr, lora_ptr, out_ptr, curr_N, K, cta_m_len, ram, slice_start_loc,
                     input_d0_stride, input_d1_stride, input_d2_stride, ls_d0_ptr, ls_d1_ptr, ls_d2_ptr,
                     output_d0_stride, output_d1_stride, BLOCK_M, BLOCK_N, BLOCK_K, SAME_STRIDE, SLICE_NUM, EVEN_K,
                     CAST_TYPE, ADD_INPUTS, USE_STRIDE_LOAD)


@triton.jit
def _lora_shrink_kernel(input_ptr, lora_ptr, out_ptr, M, N, K, token_indices_sorted_by_lora_ids, num_tokens_per_lora,
                        lora_token_start_loc, lora_ids, scaling, input_d0_stride, input_d1_stride, lora_d0_stride,
                        lora_d1_stride, lora_d2_stride, output_d0_stride, output_d1_stride, output_d2_stride,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
                        SPLIT_K: tl.constexpr, SLICE_NUM: tl.constexpr, USE_STRIDE_LOAD: tl.constexpr):

    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)

    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m = (pid_sk_m_n // SPLIT_K) % cta_m_num
    pid_n = pid_sk_m_n // (SPLIT_K * cta_m_num) % cta_n_num

    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return

    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return

    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)

    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset)

    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)

    do_shrink_kernel(pid_n, pid_sk, slice_id, lora_id, input_ptr, lora_ptr, out_ptr, N, K, cta_m_len, ram,
                     input_d0_stride, input_d1_stride, lora_d0_stride, lora_d1_stride, lora_d2_stride, output_d0_stride,
                     output_d1_stride, output_d2_stride, scaling, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, SLICE_NUM,
                     USE_STRIDE_LOAD)


@torch.inference_mode()
def lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(0) == len(lora_b_weights)
    assert output_tensor.is_contiguous()

    M = inputs.size(1)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor, lora_strides_d1_tensor, lora_strides_d2_tensor,
     hidden_sizes_tensor, same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)

    K = lora_b_weights[0].shape[-1]
    ADD_INPUTS = add_inputs
    MAX_LORAS = lora_ids.size(0)
    CAST_TYPE = False
    NUM_SLICES = len(lora_b_weights)

    if M <= 32:
        BLOCK_M = 16 if M <= 16 else 32
        NUM_WARPS = 4
    else:
        BLOCK_M = 64
        NUM_WARPS = 8
    BLOCK_N = 128
    BLOCK_K = 16
    NUM_CTAS = 1
    NUM_STAGES = 1

    EVEN_K = K % BLOCK_K == 0  # type: ignore
    if same_stride:
        use_stride_load = False
    else:
        elem_size = lora_b_weights[0].element_size()
        use_stride_load = all((w.stride(1) * elem_size) % 64 == 0 for w in lora_b_weights)

    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        NUM_SLICES,
        MAX_LORAS,
    )

    _lora_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        MAX_N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        NUM_SLICES,
        same_stride,
        use_stride_load,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
    )


@torch.inference_mode()
def lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
    scaling: float,
) -> None:
    assert no_lora_flag_cpu.numel() == 1
    if no_lora_flag_cpu.item():
        return

    assert inputs.dtype == lora_a_weights[0].dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    for weight in lora_a_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == lora_a_weights[0].size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    M = inputs.size(0)
    assert token_lora_mapping.size(0) == M
    assert token_lora_mapping.size(0) == token_indices_sorted_by_lora_ids.size(0)
    assert lora_ids.size(0) == num_tokens_per_lora.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    (lora_ptr_tensor, lora_strides_d0, lora_strides_d1,
     lora_strides_d2) = _get_lora_a_ptr(lora_a_weights, inputs.device)
    N, K = lora_a_weights[0].shape[-2:]
    NUM_SLICES = len(lora_a_weights)
    MAX_LORAS = lora_ids.size(0)

    NUM_CTAS, MAX_NREG = 1, None
    if M <= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 16, 64
        SPLIT_K, NUM_WARPS, NUM_STAGES = 8, 2, 2
    else:
        if NUM_SLICES == 1:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 16, 64
            SPLIT_K, NUM_WARPS, NUM_STAGES = 4, 4, 1
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 16, 128
            NUM_WARPS, NUM_STAGES = 8, 1
            SPLIT_K = 4 if K >= 5120 else 1

    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0  # type: ignore

    grid = (
        SPLIT_K * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        NUM_SLICES,
        MAX_LORAS,
    )

    _lora_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        NUM_SLICES,
        False,
        num_warps=NUM_WARPS,
        num_ctas=NUM_CTAS,
        num_stages=NUM_STAGES,
    )
