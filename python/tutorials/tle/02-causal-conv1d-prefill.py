# Copyright 2026- Xcoresigma Technology Co., Ltd

import torch
import numpy as np
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

import pytest
import torch.nn.functional as F
from einops import rearrange
import random

import triton
import triton.language as tl
import triton.experimental.tle as tle

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
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

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        # Handle the case if prefix caching is enabled.
        # In particular, if prefix caching is enabled, the program write additional cache states to "cache_indices_ptr"

        # Get the length of the completed sequence so far and compute the offset.
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        # Compute the offset where the first stride_block_m-aligned first full block is
        # Value in "token-space"
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        # If the sequence without the sequence_offset_index is stride_cache_chunk-aligned, then the last full chunk is the second-to-last one
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size

        # Get the number of blocks to be filled for the current sequence
        # If n_block_to_fill = 0, then only the state at the sequence end is stored
        n_block_to_fill = current_last_index - current_first_index

        # Get the index of the init block
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = (x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim)  # [BLOCK_N,]

    # cache_idx
    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                      conv_state_init_index).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (state_len <= seqlen):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None] +
                      (idx_feats * stride_x_dim)[None, :])  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                # (idx_tokens_last >= 0)[:, None] &
                (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :])  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # Compute the offset where the last block should be written in the conv_states
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_last_index).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim))[None, :] + (  # [BLOCK_N,]
                    idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            # tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) +
                                           (idx_feats * stride_conv_state_dim)[None, :] +
                                           ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                                           )  # [BLOCK_M, BLOCK_N]
                mask = ((conv_states_input_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                # tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(mask, conv_state,
                                          loaded_x)  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                                           )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None])  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None]
                          & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                                           )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)

        # Store intermediate states aligned with stride_block_m
        # The additional states are cached starting from the last stride_block_m.
        # For example:
        # If n_block_to_fill = 0, then only the state at the sequence end is cached and the process below is not involved.
        # If n_block_to_fill > 0, then the states at the sequence end and at the n_block_to_fill-last
        # stride_block_m are cached.
        # For example chunk_offset = n_block_to_fill stores the state at last_full_block
        if (chunk_offset - 1) < n_block_to_fill:
            # Store the states at the chunk boundaries from the start of the sequence
            idx_tokens_last = (last_full_block_token_index - (n_block_to_fill - chunk_offset) * B_size -
                               state_len) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (x_ptr + (idx_tokens_last * stride_x_token)[:, None] + (idx_feats * stride_x_dim)[None, :]
                      )  # [BLOCK_M,BLOCK_N,]

            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats
                                                        < dim)[None, :]  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # cache_idx
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices +
                                               current_first_index + (chunk_offset - 1)).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim))[None, :] + (  # [BLOCK_N,]
                    idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            # tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS in 2D
    # column offset: [pid_c*BLOCK_N, pid_c*BLOCK_N+1, ..., pid_c*BLOCK_N+BLOCK_N-1]
    col_offsets = idx_feats
    # row offsets: [0, dim, 2*dim, ..., (KERNEL_WIDTH-1)*dim]
    row_offsets = tl.arange(0, KERNEL_WIDTH) * stride_w_width
    w_ptrs_2d = w_ptr + row_offsets[:, None] + col_offsets[None, :]

    # Create mask for boundary conditions
    mask_w_2d = idx_feats[None, :] < dim

    w_tile_2d = tl.load(w_ptrs_2d, mask=mask_w_2d, other=0.0).to(tl.float32)
    w_tile = tl.reshape(w_tile_2d, (KERNEL_WIDTH * BLOCK_N, ))

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        # Use extract_slice to get weight columns on-demand from 1D vector
        matrix_w = tle.dsa.extract_slice(w_tile, offsets=(0 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(1 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(1 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    matrix_x = col1
                elif j == 2:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(2 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(1 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    matrix_x = col1
                elif j == 2:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(2 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    matrix_x = col2
                elif j == 3:
                    matrix_w = tle.dsa.extract_slice(w_tile, offsets=(3 * BLOCK_N, ), sizes=(BLOCK_N, ), strides=(1, ))
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
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

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)  # token-index  # feature-index
        o_ptrs = (o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token +
                  (idx_feats * stride_o_dim))

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if cache_indices are not provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block containing the initial state is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    block_size_to_align: int
        The block size to align the cached states to
    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    args = None
    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)

    x = x.transpose(0, 1).contiguous()  # (dim, seqlen) -> (seqlen, dim)
    weight = weight.transpose(0, 1).contiguous()  # (dim, width) -> (width, dim)

    out = torch.empty_like(x)
    if metadata is not None:
        nums_dict = metadata.nums_dict
        args = nums_dict
        batch_ptr = metadata.batch_ptr
        token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    else:
        seqlens = query_start_loc.diff().to("cpu")
        args = seqlens
        MAX_NUM_PROGRAMS = 1024

        batch_ptr = torch.full((MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32,
                               device=x.device)  # tracking which seq-idx the Triton program is handling
        token_chunk_offset_ptr = torch.full(
            (MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32,
            device=x.device)  # tracking BLOCK_M-based index in the sequence the Triton program is handling
    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    cu_seqlen, dim = x.shape
    width, _ = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_dim = x.stride(1)
    stride_x_token = x.stride(0)
    stride_w_dim = weight.stride(1)  # Now (width, dim), so stride(1) is for dim
    stride_w_width = weight.stride(0)  # Now (width, dim), so stride(0) is for width
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    BLOCK_M = 128
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (num_cache_lines == conv_states.shape[0] and dim == conv_states.shape[1]
                and width - 1 <= conv_states.shape[2])
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_token = out.stride(0)
        stride_o_dim = out.stride(1)
    else:
        stride_o_token = out.stride(1)
        stride_o_dim = out.stride(2)
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    # False
    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch, )
            assert conv_states is not None, ("ERROR: `has_initial_state` is used, which needs also `conv_states`")
        assert weight.stride(1) == 1
        assert (width, dim) == weight.shape  # Now (width, dim) after transpose
        assert is_channel_last, "Need to run in channel-last layout"
        if block_size_to_align is not None and block_size_to_align > 0:
            assert (block_size_to_align % BLOCK_M) == 0, ("The mamba block size needs to be divisible by the BLOCK_M")
        else:
            block_size_to_align = BLOCK_M

    if metadata is None:

        def num_program(META, seqlens):
            tot = 0

            mlist = []
            offsetlist = []  # type: ignore

            nums = -(-seqlens // META["BLOCK_M"])

            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            for idx, num in enumerate(nums):
                offsetlist.extend(range(num))  # chunk-idx if a sequence is split into multiple chunks

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
    else:

        def num_program(META, nums_dict):
            tot = nums_dict[META["BLOCK_M"]]["tot"]

            mlist = nums_dict[META["BLOCK_M"]]["mlist"]
            mlist_len = nums_dict[META["BLOCK_M"]]["mlist_len"]

            offsetlist = nums_dict[META["BLOCK_M"]]["offsetlist"]

            if nums_dict[META["BLOCK_M"]]["batch_ptr"] is not None:
                META["batch_ptr"] = nums_dict[META["BLOCK_M"]]["batch_ptr"]
                META["token_chunk_offset_ptr"] = nums_dict[META["BLOCK_M"]]["token_chunk_offset_ptr"]
            else:
                if META["batch_ptr"].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                    META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

                if META["batch_ptr"].nelement() >= mlist_len:
                    META["batch_ptr"][0:mlist_len].copy_(mlist)
                    META["token_chunk_offset_ptr"][0:mlist_len].copy_(offsetlist)
            return tot

    def grid(META):
        return (
            num_program(META, args),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
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
        block_size_to_align // BLOCK_M,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=BLOCK_M,
        BLOCK_N=512,
        num_stages=2,
    )

    return out.transpose(0, 1).to(original_x_dtype)


########################################################################################
############################# TEST Ascend Causal Conv1D ################################
########################################################################################


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


@pytest.mark.parametrize("itype", [torch.bfloat16])
@pytest.mark.parametrize("silu_activation", [True])
@pytest.mark.parametrize("has_bias", [True])
@pytest.mark.parametrize("width", [4])
@pytest.mark.parametrize("seqlen", [8, 249, 4096])
@pytest.mark.parametrize("dim", [64, 4096])
@pytest.mark.parametrize("with_padding", [True, False])
@pytest.mark.parametrize("batch", [4, 10])
def test_causal_conv1d_varlen(batch, with_padding, dim, seqlen, width, has_bias, silu_activation, itype):
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    set_random_seed(0)
    seqlens = []
    batch_size = batch
    padding = 3 if with_padding else 0
    # 4 + 3 = 7
    padded_batch_size = batch_size + padding
    # 7 - 1 = 6
    nsplits = padded_batch_size - 1

    # [0, 6，2, 3, 4, 5, 1] -> [0, 2, 3, 4, 5, 6]
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values

    # [-1, 0, 2, 3, 4, 5, 6, 7] -diff-> [1, 2, 1, 1, 1, 1, 1]
    # [[1, 2, 1, 1, 1, 1, 1]]
    seqlens.append(torch.diff(torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])).tolist())
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    # [1, 3, 4, 5, 6, 7, 8]
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    # [0, 1, 3, 4, 5, 6, 7, 8]
    cumsum = torch.concat([torch.tensor([0], dtype=torch.int32), cumsum], dim=0)

    # x.shape = [1, dim, seqlen]
    x = rearrange(
        torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=itype),
        "b s d -> b d s",
    )[:, 4096:4096 + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=itype)

    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    activation = None if not silu_activation else "silu"
    # [40, 3, 64] -> [40, 64, 3]
    final_states = torch.randn(total_entries, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)
    final_states_ref = final_states.clone()
    has_initial_states = torch.randint(0, 2, (cumsum.shape[0] - 1, ), dtype=torch.bool, device=x.device)
    # [32, 4, 7, 11]
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=x.device)[:batch_size]
    # [32, 4, 7, 11, -1, -1, -1]
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )
    out = causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=final_states,
        query_start_loc=cumsum.npu(),
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_states,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )

    out_ref = []
    out_ref_b = []

    def torch_fn(x_ref_, seqlens_, padded_state_indices_, weight_ref_, bias_ref_, activation_, final_states_ref_,
                 has_initial_states_):
        out_ref_ = []
        out_ref_b_ = []
        # 切分x的seqlen维度
        # x.shape = [1, dim, seqlen]
        # x -> splits:[(tensor0, tensor1, ... , tensor6)]
        splits = [torch.split(var, seqlens_[0], dim=-1) for var in (x_ref_)]
        for i in range(len(seqlens_[0])):  # 7次循环
            # v: (tensor0, tensor1, ... , tensor6)
            x_s = [v[i].unsqueeze(0) for v in splits][0]
            # 只计算 前batch_size 个序列
            if padded_state_indices_[i] == PAD_SLOT_ID:
                continue
            out_ref_b_.append(
                causal_conv1d_ref(
                    x_s,
                    weight_ref_,
                    bias_ref_,
                    activation=activation_,
                    return_final_states=True,
                    final_states_out=final_states_ref_[padded_state_indices_[i]].unsqueeze(0),
                    initial_states=final_states_ref_[padded_state_indices_[i]].unsqueeze(0)
                    if has_initial_states_[i] else None,
                ))
        # out_ref_b: [[out0, final_states_out0], [out1, final_states_out1], [out2, final_states_out2], [out3, final_states_out3]]
        # torch.cat: 拼接out0, out1, out2, out3在seqlen维度上
        # out_ref是个list[tensor] [out_ref_tensor]
        # out_ref_tensor.shape: [1, 64, 5]
        out_ref_.append(torch.cat([t[0] for t in out_ref_b_], dim=2))
        out_ref_tensor_ = torch.cat(out_ref_, dim=0)
        return out_ref_tensor_

    out_ref_tensor = torch_fn(x_ref, seqlens, padded_state_indices, weight_ref, bias_ref, activation, final_states_ref,
                              has_initial_states)

    assert torch.allclose(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )

    unpadded_out = out[:, :out_ref_tensor.shape[-1]]
    assert torch.allclose(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)

    triton_time = do_bench_npu(
        lambda: causal_conv1d_fn(
            x.squeeze(0),
            weight,
            bias=bias,
            conv_states=final_states,
            query_start_loc=cumsum.npu(),
            cache_indices=padded_state_indices,
            has_initial_state=has_initial_states,
            activation=activation,
            pad_slot_id=PAD_SLOT_ID,
        ), clear_l2_cache=True, collect_prof=False)
    pytorch_time = do_bench_npu(
        lambda: torch_fn(x_ref, seqlens, padded_state_indices, weight_ref, bias_ref, activation, final_states_ref,
                         has_initial_states), clear_l2_cache=True, collect_prof=False)
    print("x.shape =", x.shape)
    print("with_padding =", with_padding)
    print("width =", width)
    print("has_bias =", has_bias)
    print("silu_activation =", silu_activation)
    print(f"[Triton  Casual Conv1d Prefill] Time: {triton_time:.4f} us")
    print(f"[Pytorch Casual Conv1d Prefill] Time: {pytorch_time:.4f} us")
    print("[PASSED]")


if __name__ == "__main__":
    print(torch_npu.__version__)
    test_causal_conv1d_varlen(batch=4, with_padding=True, dim=64, seqlen=8, width=4, has_bias=True,
                              silu_activation=True, itype=torch.bfloat16)
    test_causal_conv1d_varlen(batch=8, with_padding=True, dim=4096, seqlen=6, width=3, has_bias=False,
                              silu_activation=True, itype=torch.bfloat16)
    test_causal_conv1d_varlen(batch=8, with_padding=True, dim=2048, seqlen=1024, width=4, has_bias=False,
                              silu_activation=True, itype=torch.bfloat16)
