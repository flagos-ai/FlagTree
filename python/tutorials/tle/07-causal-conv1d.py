"""
Causal Conv1d with TLE extract_tile
====================================

Compares a Triton baseline causal_conv1d kernel against a TLE optimized
version that uses ``tle.extract_tile``.

Both varlen forward and update (step/update) are included.

Usage
-----
    # correctness only (default)
    python python/tutorials/tle/07-causal-conv1d.py

    # correctness + benchmark table
    python python/tutorials/tle/07-causal-conv1d.py --benchmark

    # specify dtype
    python python/tutorials/tle/07-causal-conv1d.py --benchmark --dtype float16
"""

# %%
# Setup
# -----

import argparse

import numpy as np
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PAD_SLOT_ID = -1


def _print_env():
    """Print test environment information for reproducibility."""
    print(f"GPU: {torch.cuda.get_device_name()} | CUDA: {torch.version.cuda} | Triton: {triton.__version__}")
    print()


# Benchmark parameters
BENCH_WARMUP = 10
BENCH_REP = 100
BENCH_RUNS = 5  # number of runs for stddev

# %%
# Kernels (baseline v1)
# ---------------------


@triton.jit()
def _causal_conv1d_fwd_kernel_v1(
    x_ptr,
    w_ptr,
    bias_ptr,
    initial_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    o_ptr,
    dim: tl.constexpr,
    seqlen: tl.int32,
    num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_block_m: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        n_block_to_fill = current_last_index - current_first_index

        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = (x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim)

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                      conv_state_init_index).to(tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    conv_states_base = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))

    w_base = w_ptr + (idx_feats * stride_w_dim)

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                col0 = tl.load(prior_tokens, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                col1 = tl.load(prior_tokens, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                col2 = tl.load(prior_tokens, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                col2 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
                col1 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
                col0 = tl.load(prior_tokens - 3 * stride_conv_state_tok, mask_w, 0.0)
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] +
                      (idx_feats * stride_x_dim)[None, :])
            mask_x = ((idx_tokens_last >= 0)[:, None]
                      & (idx_tokens_last < seqlen)[:, None]
                      & (idx_feats < dim)[None, :])
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_last_index).to(tl.int64)

            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) +
                                       (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv *
                                                                                        stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)

                conv_states_ptrs_source = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                                           (idx_feats * stride_conv_state_dim)[None, :] +
                                           ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None])
                mask = ((conv_states_input_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen
                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)

                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None])
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None])
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            col0 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            col1 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            col2 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col1 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
            col0 = tl.load(prior_tokens - 3 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")

        if (chunk_offset - 1) < n_block_to_fill:
            idx_tokens_last = (last_full_block_token_index -
                               (n_block_to_fill - chunk_offset) * B_size - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :])
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_first_index + (chunk_offset - 1)).to(tl.int64)

            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) +
                                       (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv *
                                                                                        stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token +
                  (idx_feats * stride_o_dim))
        tl.store(o_ptrs, acc, mask=mask_1d)


@triton.jit()
def _causal_conv1d_update_kernel_v1(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    block_idx_last_scheduled_token,
    initial_state_idx,
    o_ptr,
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init).to(
        tl.int64)

    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = (tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1)
    else:
        conv_state_token_offset = 0

    conv_states_base = (conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        col3 = tl.load(prior_tokens + 3 * stride_conv_state_tok, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        col4 = tl.load(prior_tokens + 4 * stride_conv_state_tok, mask_w, 0.0)

    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                              conv_state_token_offset * stride_conv_state_tok +
                              (idx_feats * stride_conv_state_dim)[None, :] +
                              ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[:, None])
    mask = ((conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :])
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)

    x_ptrs = (x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None])
    mask_x = ((idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :])
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index).to(
        tl.int64)
    conv_state_ptrs_target = (conv_state_ptr + (conv_states_offset * stride_conv_state_seq) +
                              (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens * stride_conv_state_tok)[:,
                                                                                                                   None]
    mask_out = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_out)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    w_base = w_ptr + (idx_feats * stride_w_dim)
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
        w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + (4 * stride_w_width), mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + (5 * stride_w_width), mask_w, other=0.0)

    x_base_1d = x_base
    mask_x_1d = idx_feats < dim

    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)

            acc += matrix_x * matrix_w

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim))
        tl.store(o_ptrs, acc, mask=mask_1d)


# %%
# Kernels (TLE v2)
# ----------------
@triton.jit()
def _causal_conv1d_fwd_kernel_v2(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # (batch, n_blocks + padding) The second dimension contains
    # the block indices relevant for each sequence
    # plus potential 0-padding at the beginning and at the end
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,  # (batch,)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    num_computed_tokens,  # (batch,)
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (KERNEL_WIDTH - 1)  # can be passed via argument if it's not the same as this value

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        n_block_to_fill = current_last_index - current_first_index

        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = (x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim)  # [BLOCK_N,]

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                      conv_state_init_index).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            return

    conv_states_base = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # x_tile_start = token_offset - (KERNEL_WIDTH - 1)
    # # idx_rows = tl.arange(0, X_TILE_M_P2)
    # idx_rows = tl.arange(0, triton.next_power_of_2(BLOCK_M + KERNEL_WIDTH - 1))
    # offs_tok = x_tile_start + idx_rows
    # tile_ptrs = (
    #         x_ptr
    #         + (sequence_start_index + offs_tok)[:, None] * stride_x_token
    #         + (idx_feats * stride_x_dim)[None, :]
    #     )
    # tile_mask = (
    #         (offs_tok >= 0)[:, None]
    #         & (offs_tok < seqlen)[:, None]
    #         & (idx_feats < dim)[None, :]
    #         & (idx_rows < (BLOCK_M + KERNEL_WIDTH - 1))[:, None]
    #     )
    # tile_data = tl.load(
    #         tile_ptrs, mask=tile_mask, other=0.0, cache_modifier=".ca"
    #     )

    if chunk_offset == 0:
        chunk = 0
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        mask_w = idx_feats < dim
        # ----------------------------------------------------------------
        # tile_data load at chunk_offset == 0:
        # read from token_offset (0), without shifting left by KERNEL_WIDTH - 1
        # ----------------------------------------------------------------
        idx_rows = tl.arange(0, triton.next_power_of_2(BLOCK_M + KERNEL_WIDTH - 1))
        offs_tok = token_offset + idx_rows
        tile_ptrs = (x_ptr + (sequence_start_index + offs_tok)[:, None] * stride_x_token +
                     (idx_feats * stride_x_dim)[None, :])
        tile_mask = ((offs_tok >= 0)[:, None]
                     & (offs_tok < seqlen)[:, None]
                     & (idx_feats < dim)[None, :]
                     & (idx_rows < (BLOCK_M + KERNEL_WIDTH - 1))[:, None])
        tile_data = tl.load(tile_ptrs, mask=tile_mask, other=0.0, cache_modifier=".ca")

        if load_init_state:
            # ----------------------------------------------------------------
            # [extract_tile] Load entire conv_state window in one tile load:
            # Before: KERNEL_WIDTH-1 separate scalar loads per column
            # After: 1 tile load [NP2_STATELEN, BLOCK_N], then slice with extract_tile
            # ----------------------------------------------------------------
            # Broadcast: conv_states_base[None, :]   [1, BLOCK_N]
            #          + idx_state[:, None] * stride  [NP2_STATELEN, 1]
            #          => [NP2_STATELEN, BLOCK_N]
            idx_state = tl.arange(0, NP2_STATELEN)
            conv_state_ptrs_full = (conv_states_base[None, :] + idx_state[:, None] * stride_conv_state_tok
                                    )  # [NP2_STATELEN, BLOCK_N]
            mask_state_full = (
                (idx_state[:, None] < state_len) & (idx_feats[None, :] < dim))  # bool mask [NP2_STATELEN, BLOCK_N]
            state_tile = tl.load(conv_state_ptrs_full, mask_state_full, other=0.0)  # [NP2_STATELEN, BLOCK_N]

            if KERNEL_WIDTH == 2:
                col0 = tle.extract_tile(state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
            if KERNEL_WIDTH == 3:
                col1 = tle.extract_tile(state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
                col0 = tle.extract_tile(state_tile, index=[state_len - 2, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
            if KERNEL_WIDTH == 4:
                col2 = tle.extract_tile(state_tile, index=[state_len - 1, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
                col1 = tle.extract_tile(state_tile, index=[state_len - 2, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
                col0 = tle.extract_tile(state_tile, index=[state_len - 3, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
            if KERNEL_WIDTH == 5:
                col2 = tle.extract_tile(state_tile, index=[state_len - 2, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
                col1 = tle.extract_tile(state_tile, index=[state_len - 3, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
                col0 = tle.extract_tile(state_tile, index=[state_len - 4, 0], tile_shape=[1, BLOCK_N]).reshape(
                    (BLOCK_N, ))
        else:
            if KERNEL_WIDTH >= 2:
                col0 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:
                col1 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:
                col2 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
        # STEP 2: write back conv_state
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] +
                      (idx_feats * stride_x_dim)[None, :])  # [NP2_STATELEN, BLOCK_N]
            mask_x = ((idx_tokens_last >= 0)[:, None]
                      & (idx_tokens_last < seqlen)[:, None]
                      & (idx_feats < dim)[None, :])
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # [NP2_STATELEN, BLOCK_N]

            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_last_index).to(tl.int64)

            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) +
                                       (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv *
                                                                                        stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                # Shift right by seqlen to read the remaining cached history
                conv_states_ptrs_source = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                                           (idx_feats * stride_conv_state_dim)[None, :] +
                                           ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                                           )  # [NP2_STATELEN, BLOCK_N]
                mask = ((conv_states_input_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen
                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # [NP2_STATELEN, BLOCK_N]
                # Merge cached history (where mask is True) with new token data
                # (where mask is False) to form the updated conv_state.

                # NOTE: insert_tile approach was explored as an alternative to
                # tl.where + tl.debug_barrier. The loop below shows the intended
                # pattern — build new_conv_state row by row using extract_tile /
                # insert_tile, avoiding the compiler bug in tl.where entirely.
                # Currently kept as a comment for future reference; the code
                # still uses tl.where with the required debug_barrier workaround.

                tl.debug_barrier()  # required: tl.where bug when result of another tl.load
                # Where mask is True: take cached history (left-shifted).
                # Where mask is False: take new tokens from x.
                new_conv_state = tl.where(mask, conv_state,
                                          loaded_x)  # BUG in 'tl.where' requires a barrier before this

                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                                           )  # [NP2_STATELEN, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

            else:  # load_init_state == False
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None])
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        load_init_state = True
        chunk = 1
        x_tile_start = token_offset - (KERNEL_WIDTH - 1)
        idx_rows = tl.arange(0, triton.next_power_of_2(BLOCK_M + KERNEL_WIDTH - 1))
        # idx_rows = tl.arange(0, X_TILE_M_P2)
        offs_tok = x_tile_start + idx_rows
        tile_ptrs = (x_ptr + (sequence_start_index + offs_tok)[:, None] * stride_x_token +
                     (idx_feats * stride_x_dim)[None, :])
        tile_mask = ((offs_tok >= 0)[:, None]
                     & (offs_tok < seqlen)[:, None]
                     & (idx_feats < dim)[None, :]
                     & (idx_rows < (BLOCK_M + KERNEL_WIDTH - 1))[:, None])
        tile_data = tl.load(tile_ptrs, mask=tile_mask, other=0.0, cache_modifier=".ca")

        if KERNEL_WIDTH == 2:
            col0 = tl.reshape(
                tle.extract_tile(tile_data, index=[0, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
        if KERNEL_WIDTH == 3:
            col0 = tl.reshape(
                tle.extract_tile(tile_data, index=[0, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
            col1 = tl.reshape(
                tle.extract_tile(tile_data, index=[1, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
        if KERNEL_WIDTH == 4:
            col0 = tl.reshape(
                tle.extract_tile(tile_data, index=[0, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
            col1 = tl.reshape(
                tle.extract_tile(tile_data, index=[1, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
            col2 = tl.reshape(
                tle.extract_tile(tile_data, index=[2, 0], tile_shape=[1, BLOCK_N]),
                [BLOCK_N],
            )
        if (chunk_offset - 1) < n_block_to_fill:
            idx_tokens_last = (last_full_block_token_index -
                               (n_block_to_fill - chunk_offset) * B_size - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :])
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[None, :]
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)

            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_first_index + (chunk_offset - 1)).to(tl.int64)

            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq) +
                                       (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv *
                                                                                        stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.reshape(
                        tle.extract_tile(
                            tile_data,
                            index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                            tile_shape=[1, BLOCK_N],
                        ),
                        [BLOCK_N],
                    )
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.reshape(
                        tle.extract_tile(
                            tile_data,
                            index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                            tile_shape=[1, BLOCK_N],
                        ),
                        [BLOCK_N],
                    )
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.reshape(
                        tle.extract_tile(
                            tile_data,
                            index=[idx_token + chunk * (KERNEL_WIDTH - 1), 0],
                            tile_shape=[1, BLOCK_N],
                        ),
                        [BLOCK_N],
                    )
            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token +
                  (idx_feats * stride_o_dim))
        tl.store(o_ptrs, acc, mask=mask_1d)


@triton.jit()
def _causal_conv1d_update_kernel_v2(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init).to(
        tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        conv_state_token_offset = (tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1)
    else:
        conv_state_token_offset = 0

    # STEP 1: load conv_state tile and extract columns via extract_tile
    conv_states_base = (conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))
    mask_w = idx_feats < dim

    idx_state = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_full = (conv_states_base[None, :] +
                            (conv_state_token_offset + idx_state)[:, None] * stride_conv_state_tok
                            )  # [NP2_STATELEN, BLOCK_N]
    mask_state_full = (idx_state[:, None] < state_len) & (idx_feats[None, :] < dim)
    state_tile = tl.load(conv_state_ptrs_full, mask_state_full, other=0.0)  # [NP2_STATELEN, BLOCK_N]

    if KERNEL_WIDTH >= 2:
        col0 = tle.extract_tile(state_tile, index=[0, 0], tile_shape=[1, BLOCK_N]).reshape((BLOCK_N, ))
    if KERNEL_WIDTH >= 3:
        col1 = tle.extract_tile(state_tile, index=[1, 0], tile_shape=[1, BLOCK_N]).reshape((BLOCK_N, ))
    if KERNEL_WIDTH >= 4:
        col2 = tle.extract_tile(state_tile, index=[2, 0], tile_shape=[1, BLOCK_N]).reshape((BLOCK_N, ))
    if KERNEL_WIDTH >= 5:
        col3 = tle.extract_tile(state_tile, index=[3, 0], tile_shape=[1, BLOCK_N]).reshape((BLOCK_N, ))
    if KERNEL_WIDTH >= 6:
        col4 = tle.extract_tile(state_tile, index=[4, 0], tile_shape=[1, BLOCK_N]).reshape((BLOCK_N, ))

    # STEP 2: build updated conv_state and write back
    idx_tokens = tl.arange(0, NP2_STATELEN)

    conv_state_ptrs_source = (conv_state_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                              conv_state_token_offset * stride_conv_state_tok +
                              (idx_feats * stride_conv_state_dim)[None, :] +
                              ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[:, None]
                              )  # [NP2_STATELEN, BLOCK_N]
    mask = ((conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :])
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]
    x_ptrs = (x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None])  # [NP2_STATELEN, BLOCK_N]
    mask_x = ((idx_tokens - VAL >= 0)[:, None] & (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :])
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)  # [NP2_STATELEN, BLOCK_N]

    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_states_offset = tl.load(conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index).to(
        tl.int64)
    conv_state_ptrs_target = (conv_state_ptr + (conv_states_offset * stride_conv_state_seq) +
                              (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens * stride_conv_state_tok)[:,
                                                                                                                   None]
    mask_out = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_out)

    # STEP 3: initialize accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # STEP 4: preload weights
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: per-token convolution
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim)
        o_ptrs = (o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim))
        tl.store(o_ptrs, acc, mask=mask_1d)


# %%
# Python wrappers
# ---------------


def _launch_fwd_kernel(kernel_fn, x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state,
                       activation, pad_slot_id):
    if isinstance(activation, bool) and activation:
        activation = "silu"

    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)

    seqlens = query_start_loc.diff().to("cpu")
    MAX_NUM_PROGRAMS = 1024
    batch_ptr = torch.full((MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32, device=x.device)
    token_chunk_offset_ptr = torch.full((MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32, device=x.device)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    BLOCK_M = 8
    num_cache_lines = conv_states.size(0)

    stride_x_dim, stride_x_token = x.stride()
    stride_w_dim, stride_w_width = weight.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_states.stride()
    stride_o_dim, stride_o_token = out.stride()
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    def num_program(META, seqlens):
        nums = -(-seqlens // META["BLOCK_M"])
        tot = nums.sum().item()
        mlist = np.repeat(np.arange(len(nums)), nums)
        offsetlist = []
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))

        if META["batch_ptr"].nelement() < len(mlist):
            newlen = len(mlist) + 1
            META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
            META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

        if META["batch_ptr"].nelement() >= len(mlist):
            META["batch_ptr"][0:len(mlist)].copy_(torch.from_numpy(np.array(mlist)))
            META["token_chunk_offset_ptr"][0:len(mlist)].copy_(torch.from_numpy(np.array(offsetlist)))

        META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
        META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(META["x_ptr"].device)
        return tot

    def grid(META):
        return (num_program(META, seqlens), triton.cdiv(dim, META["BLOCK_N"]))

    kernel_fn[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        None,
        None,
        None,
        None,
        out,
        dim,
        cu_seqlen,
        num_cache_lines,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        stride_o_dim,
        stride_o_token,
        1,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=False,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        num_stages=2,
    )
    return out.to(original_x_dtype)


def causal_conv1d_fn_v1(x, weight, bias, conv_states, query_start_loc, cache_indices=None, has_initial_state=None,
                        activation="silu", pad_slot_id=PAD_SLOT_ID):
    return _launch_fwd_kernel(_causal_conv1d_fwd_kernel_v1, x, weight, bias, conv_states, query_start_loc,
                              cache_indices, has_initial_state, activation, pad_slot_id)


def causal_conv1d_fn_v2(x, weight, bias, conv_states, query_start_loc, cache_indices=None, has_initial_state=None,
                        activation="silu", pad_slot_id=PAD_SLOT_ID):
    return _launch_fwd_kernel(_causal_conv1d_fwd_kernel_v2, x, weight, bias, conv_states, query_start_loc,
                              cache_indices, has_initial_state, activation, pad_slot_id)


def _launch_update_kernel(kernel_fn, x, conv_state, weight, bias, activation, conv_state_indices, pad_slot_id):
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    num_cache_lines, _, state_len = conv_state.size()
    out = x

    stride_w_dim, stride_w_width = weight.stride()
    stride_x_seq, stride_x_dim, stride_x_token = x.stride()
    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = conv_state_indices.stride(0) if conv_state_indices is not None else 0
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (batch, triton.cdiv(dim, META["BLOCK_N"]))

    kernel_fn[grid](
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        None,
        None,
        None,
        None,
        out,
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=False,
        IS_APC_ENABLED=False,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


def causal_conv1d_update_v1(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                            pad_slot_id=PAD_SLOT_ID):
    return _launch_update_kernel(_causal_conv1d_update_kernel_v1, x, conv_state, weight, bias, activation,
                                 conv_state_indices, pad_slot_id)


def causal_conv1d_update_v2(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
                            pad_slot_id=PAD_SLOT_ID):
    return _launch_update_kernel(_causal_conv1d_update_kernel_v2, x, conv_state, weight, bias, activation,
                                 conv_state_indices, pad_slot_id)


# %%
# Correctness check
# -----------------


def _make_varlen_data(dim, total_seqlen, batch, width, dtype):
    torch.manual_seed(0)
    eos_pos = torch.randperm(total_seqlen - 1)[:batch - 1].sort().values
    seqlens = torch.diff(
        torch.cat([
            torch.tensor([-1], dtype=torch.int32),
            eos_pos.to(dtype=torch.int32),
            torch.tensor([total_seqlen - 1], dtype=torch.int32),
        ]))
    query_start_loc = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(seqlens, dim=0).to(torch.int32),
    ]).to(DEVICE)

    x = torch.randn(dim, total_seqlen, device=DEVICE, dtype=dtype)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    conv_states = torch.randn(batch, dim, width - 1, device=DEVICE, dtype=dtype)
    cache_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=DEVICE)
    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def _make_update_data(dim, batch, width, dtype):
    torch.manual_seed(0)
    x = torch.randn(batch, dim, 1, device=DEVICE, dtype=dtype)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    conv_state = torch.randn(batch, dim, width - 1, device=DEVICE, dtype=dtype)
    conv_state_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    return x, weight, bias, conv_state, conv_state_indices


def run_correctness(dim=1024, total_seqlen=2048, batch=32, width=4, dtype=torch.bfloat16):
    # varlen forward
    x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state = \
        _make_varlen_data(dim, total_seqlen, batch, width, dtype)

    out_v1 = causal_conv1d_fn_v1(x.clone(), weight, bias, conv_states.clone(), query_start_loc, cache_indices,
                                 has_initial_state, activation="silu")
    out_v2 = causal_conv1d_fn_v2(x.clone(), weight, bias, conv_states.clone(), query_start_loc, cache_indices,
                                 has_initial_state, activation="silu")
    torch.testing.assert_close(out_v1.float(), out_v2.float(), rtol=1e-2, atol=1e-2)
    print(f"  pass  varlen dim={dim} seqlen={total_seqlen} batch={batch} width={width} {dtype}")

    # update
    x_u, weight_u, bias_u, conv_state_u, conv_state_indices_u = \
        _make_update_data(dim, batch, width, dtype)

    out_u_v1 = causal_conv1d_update_v1(x_u.clone(), conv_state_u.clone(), weight_u, bias_u, activation="silu",
                                       conv_state_indices=conv_state_indices_u)
    out_u_v2 = causal_conv1d_update_v2(x_u.clone(), conv_state_u.clone(), weight_u, bias_u, activation="silu",
                                       conv_state_indices=conv_state_indices_u)
    torch.testing.assert_close(out_u_v1.float(), out_u_v2.float(), rtol=1e-2, atol=1e-2)
    print(f"  pass  update dim={dim} batch={batch} width={width} {dtype}")


# %%
# Benchmark
# ---------


def _bench_one(fn, warmup=BENCH_WARMUP, rep=BENCH_REP, runs=BENCH_RUNS):
    """Run do_bench multiple times with quantiles, return stats dict."""
    p50s, p20s, p80s = [], [], []
    for _ in range(runs):
        ms, p80, p20 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
        p50s.append(ms if isinstance(ms, float) else ms[0])
        p20s.append(p20 if isinstance(p20, float) else p20[0])
        p80s.append(p80 if isinstance(p80, float) else p80[0])
    p50s = np.array(p50s)
    return {
        "mean": float(p50s.mean()),
        "std": float(p50s.std()),
        "p50": float(np.median(p50s)),
        "p90": float(np.percentile(p50s, 90)),
        "min": float(np.min(p20s)),
        "max": float(np.max(p80s)),
    }


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--total_seqlen", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "fp16", "bf16"])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(argv)

    dtype = torch.bfloat16 if "bf" in args.dtype else torch.float16

    _print_env()
    run_correctness(args.dim, args.total_seqlen, args.batch, args.width, dtype)

    if args.benchmark:
        # ------------------------------------------------------------------
        # You can sweep more configs here by editing the lists below.
        # Each (dim, batch) pair produces one benchmark table.
        # ------------------------------------------------------------------
        VARLEN_CONFIGS = [(4096, 32), (8192, 128)]
        UPDATE_CONFIGS = [(256, ), (1024, )]  # batch values
        for dim, batch in VARLEN_CONFIGS:
            _run_benchmark_varlen(dim, batch, dtype)
        for batch in UPDATE_CONFIGS:
            _run_benchmark_update(batch[0], dtype)


def _run_benchmark_varlen(dim, batch, dtype):
    width = 4
    x_vals = [2048, 4096, 8192, 16384]
    print(
        f"\n--- Varlen | dim={dim} batch={batch} width={width} {dtype} | Warmup={BENCH_WARMUP} Rep={BENCH_REP} Runs={BENCH_RUNS} ---"
    )
    print()
    print(
        f"{'seqlen':<10} {'Baseline mean':<14} {'p50':<12} {'p90':<12} {'TLE mean':<14} {'p50':<12} {'p90':<12} {'Speedup':<10} {'correctness':<12}"
    )

    for total_seqlen in x_vals:
        x, w, b, cs, qsl, ci, his = _make_varlen_data(dim, total_seqlen, batch, width, dtype)
        s_b = _bench_one(lambda: causal_conv1d_fn_v1(x, w, b, cs, qsl, ci, his, "silu"))
        s_t = _bench_one(lambda: causal_conv1d_fn_v2(x, w, b, cs, qsl, ci, his, "silu"))
        sp = s_b["p50"] / s_t["p50"] if s_t["p50"] > 0 else 1.0
        out_v1 = causal_conv1d_fn_v1(x.clone(), w, b, cs.clone(), qsl, ci, his, "silu")
        out_v2 = causal_conv1d_fn_v2(x.clone(), w, b, cs.clone(), qsl, ci, his, "silu")
        try:
            torch.testing.assert_close(out_v1.float(), out_v2.float(), rtol=1e-2, atol=1e-2)
            chk = "pass"
        except Exception:
            chk = "FAIL"
        sp_str = f"{sp:.2f}x"
        print(f"{total_seqlen:<10} {s_b['mean']:<14.4f} {s_b['p50']:<12.4f} {s_b['p90']:<12.4f} "
              f"{s_t['mean']:<14.4f} {s_t['p50']:<12.4f} {s_t['p90']:<12.4f} {sp_str:<10} {chk}")


def _run_benchmark_update(batch, dtype):
    width = 4
    x_vals = [1024, 2048, 4096, 8192]
    print(
        f"\n--- Update | batch={batch} dim=[1024,2048,4096,8192] width={width} {dtype} | Warmup={BENCH_WARMUP} Rep={BENCH_REP} Runs={BENCH_RUNS} ---"
    )
    print()
    print(
        f"{'dim':<8} {'Baseline mean':<14} {'p50':<12} {'p90':<12} {'TLE mean':<14} {'p50':<12} {'p90':<12} {'Speedup':<10} {'correctness':<12}"
    )

    for dim in x_vals:
        x, w, b, cs, ci = _make_update_data(dim, batch, width, dtype)
        s_b = _bench_one(lambda: causal_conv1d_update_v1(x, cs, w, b, "silu", ci))
        s_t = _bench_one(lambda: causal_conv1d_update_v2(x, cs, w, b, "silu", ci))
        sp = s_b["p50"] / s_t["p50"] if s_t["p50"] > 0 else 1.0
        # re-generate data for correctness check (benchmark calls modified x/cs in place)
        x2, _, _, cs2, _ = _make_update_data(dim, batch, width, dtype)
        cs_v1 = cs2.clone()
        cs_v2 = cs2.clone()
        out_v1 = causal_conv1d_update_v1(x2.clone(), cs_v1, w, b, "silu", ci)
        out_v2 = causal_conv1d_update_v2(x2.clone(), cs_v2, w, b, "silu", ci)
        try:
            torch.testing.assert_close(out_v1.float(), out_v2.float(), rtol=1e-2, atol=1e-2)
            chk = "pass"
        except Exception:
            chk = "FAIL"
        sp_str = f"{sp:.2f}x"
        print(f"{dim:<8} {s_b['mean']:<14.4f} {s_b['p50']:<12.4f} {s_b['p90']:<12.4f} "
              f"{s_t['mean']:<14.4f} {s_t['p50']:<12.4f} {s_t['p90']:<12.4f} {sp_str:<10} {chk}")


if __name__ == "__main__":
    main()
