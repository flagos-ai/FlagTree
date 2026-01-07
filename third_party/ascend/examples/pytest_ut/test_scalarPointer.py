import pytest
import os
import sys
import math

import pytest
import os
import sys
import math

current_file = os.path.abspath(__file__)
vllm_root_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, vllm_root_path)

#from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd_kernel
from einops import rearrange
import torch
import torch_npu
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    final_states_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    # Matrix dimensions
    dim,
    nchunks,
    seqlen,
    chunk_size,
    # Strides
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_final_states_batch,
    stride_final_states_head,
    stride_final_states_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_initstates_batch,
    stride_initstates_head,
    stride_initstates_dim,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_CONT_BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # - states will be the past state of the sequence that continues on the current check
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        #dA_cs = tl.load(dA_cs_ptr + tl.arange(0,1)).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk



def _state_passing_fwd(
    states,
    dA_chunk_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        if is_cont_batched:
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            assert seq_idx is not None, ""
        else:
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim),
                      device=states.device,
                      dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim),
                               device=states.device,
                               dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    # with torch.cuda.device(states.device.index):
    _state_passing_fwd_kernel[grid](
        states,
        out,
        final_states,
        dA_chunk_cumsum,
        initial_states,
        seq_idx,
        dim,
        nchunks,
        seqlen if seq_idx is not None else 0,
        chunk_size if seq_idx is not None else 0,
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        final_states.stride(0),
        final_states.stride(1),
        final_states.stride(2),
        dA_chunk_cumsum.stride(0),
        dA_chunk_cumsum.stride(2),
        dA_chunk_cumsum.stride(1),
        *((initial_states.stride(0), initial_states.stride(1),
           initial_states.stride(2)) if initial_states is not None else
          (0, 0, 0)),
        *((seq_idx.stride(0),
           seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        HAS_INITSTATES=initial_states is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_CONT_BATCHED=is_cont_batched,
    )
    return out, final_states

@pytest.mark.perf(repeat=17)
def test_state_passing_fwd():
    states = torch.randn((1, 7, 3, 5, 5), dtype=torch.float32).npu()
    dA_cumsum = torch.randn((1, 3, 7, 17), dtype=torch.float32).npu()
    initial_states = None
    cu_seqlens = None
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None else None,
        seq_idx=None,
        chunk_size=17,
        out_dtype=torch.float32,
        is_cont_batched=cu_seqlens is not None)

