"""
HSTU Attention
===============
"""

from typing import List, Optional, Tuple
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver
import numpy as np
import torch.nn.functional as F

DEVICE = "npu"
BLOCK_FWD = 64
BLOCK_BWD = 32


def get_npu_properties(coreType):
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)[coreType]


@triton.jit
def _hstu_attn_fwd_one_block(
    q,
    k_block_ptr,
    v_block_ptr,
    bias_block_ptr,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    mask_block,
):
    k = tl.load(k_block_ptr)
    qk = tl.dot(q, tl.trans(k)) * alpha
    if HAS_BIAS:
        rel_attn_bias = tl.load(bias_block_ptr)
        qk = qk + rel_attn_bias
    silu = qk / (1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        silu = tl.where(mask_block, silu, 0)
    v = tl.load(v_block_ptr)
    silu = silu.to(v.dtype)
    return tl.dot(silu, v)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    Out,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    alpha,
    head_num,
    MAX_SEQ_LEN,
    off_batch,
    off_head,
    start_m,
    seq_start,
    seq_len,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mask_block,
    bias,
):
    off_head = off_head.to(tl.int64)
    off_seq = seq_start.to(tl.int64)
    start_m = start_m.to(tl.int32)

    # initialize offsets
    q_offset = off_seq * stride_qm + off_head * stride_qh
    k_offset = off_seq * stride_kn + off_head * stride_kh
    v_offset = off_seq * stride_vn + off_head * stride_kh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seq_len, BLOCK_D_Q),
        strides=(stride_qm, 1),
        offsets=(start_m, 0),
        block_shape=(BLOCK_M, BLOCK_D_Q),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(seq_len, BLOCK_D_Q),
        strides=(stride_kn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D_Q),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seq_len, BLOCK_D_V),
        strides=(stride_vn, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D_V),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr)

    acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
    if CAUSAL:
        low = 0
        high = start_m + BLOCK_M
    else:
        low = 0
        high = seq_len

    bias_block_ptr = None
    if HAS_BIAS:
        bias_block_ptr = tl.make_block_ptr(
            base=bias + off_batch * head_num * MAX_SEQ_LEN * MAX_SEQ_LEN + off_head * MAX_SEQ_LEN * MAX_SEQ_LEN,
            shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
            strides=(MAX_SEQ_LEN, 1),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

    for start_n in range(low, high, BLOCK_N):
        acc += _hstu_attn_fwd_one_block(
            q=q,
            k_block_ptr=k_block_ptr,
            v_block_ptr=v_block_ptr,
            bias_block_ptr=bias_block_ptr,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL and start_m == start_n,
            HAS_BIAS=HAS_BIAS,
            mask_block=mask_block,
        )
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
        if HAS_BIAS:
            bias_block_ptr = tl.advance(bias_block_ptr, (0, BLOCK_N))

    # rematerialize offsets to save registers
    offs_v_d = tl.arange(0, BLOCK_D_V)
    off_o = Out + off_seq * stride_om + off_head * stride_oh
    offs_m = start_m + tl.arange(0, BLOCK_M)
    out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    Out,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    alpha: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    head_dim: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CORE_NUM: tl.constexpr,
    tasks: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    mask,
    bias,
):
    core_id = tl.program_id(0)
    cur_batch = 0
    mask_block = None
    if CAUSAL and mask is not None:
        mask_ptr = tl.make_block_ptr(
            base=mask,
            shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
            strides=(MAX_SEQ_LEN, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_M),
            order=(1, 0),
        )
        mask_block = tl.load(mask_ptr)
    for col in range(core_id, tasks, CORE_NUM):
        seq_end = tl.load(seq_offsets + cur_batch + 1)
        start_m = col * BLOCK_M
        while start_m >= seq_end * head_num // 2:
            cur_batch += 1
            seq_end = tl.load(seq_offsets + cur_batch + 1)
        seq_start = tl.load(seq_offsets + cur_batch)
        seq_len = seq_end - seq_start
        off_batch = cur_batch
        off_head = (start_m - seq_start * head_num // 2) // (seq_len // 2)
        start_m_1 = (start_m - seq_start * head_num // 2) % (seq_len // 2)
        start_m_2 = seq_len - start_m_1 - BLOCK_M
        _hstu_attn_fwd_compute(Q, K, V, seq_offsets, Out, stride_qm, stride_qh, stride_kn, stride_kh,
            stride_vn, stride_vh, stride_om, stride_oh, alpha, head_num, MAX_SEQ_LEN, off_batch, off_head,
            start_m_1, seq_start, seq_len, CAUSAL, HAS_BIAS, head_dim, head_dim, BLOCK_M, BLOCK_N,
            mask_block=mask_block,
            bias=bias,
        )
        _hstu_attn_fwd_compute(Q, K, V, seq_offsets, Out, stride_qm, stride_qh, stride_kn, stride_kh,
            stride_vn, stride_vh, stride_om, stride_oh, alpha, head_num, MAX_SEQ_LEN, off_batch, off_head,
            start_m_2, seq_start, seq_len, CAUSAL, HAS_BIAS, head_dim, head_dim, BLOCK_M, BLOCK_N,
            mask_block=mask_block,
            bias=bias,
        )


@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs,
    dq_ptrs,
    mask_n,
    do_ptrs,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    max_ids,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias_block_ptr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    # recompute qk and silu
    q = tl.load(
        q_ptrs + start_m * stride_qm,
        mask=mask_m[:, None],
        other=0.0,
    )
    q_trans = tl.trans(q)
    qk_trans = tl.dot(k, q_trans) * alpha
    if HAS_BIAS:
        rel_attn_bias = tl.load(bias_block_ptr)
        qk_trans = qk_trans + tl.trans(rel_attn_bias)
    sig_trans = 1.0 / (1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
        pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
        invalid_mask_trans = invalid_mask_trans | (pos_offs_m_minus_n > 0)
        silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    )
    dv += tl.dot(silu_trans, do)
    # compute dk and dq  (dqk = do * v^T  dk = dqk^T * q  dq = dqk * k)
    dqk_trans = tl.dot(v, tl.trans(do))
    dqk_trans = dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    if CAUSAL:
        dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)
    dq = tl.load(
        dq_ptrs + start_m * stride_dqm,
        mask=mask_m[:, None],
        other=0.0,
    )
    dq += tl.dot(tl.trans(dqk_trans), k) * alpha
    tl.store(
        dq_ptrs + start_m * stride_dqm,
        dq,
        mask=mask_m[:, None],
    )
    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, q)
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if CAUSAL:
        low = start_n
        high = seq_len
    else:
        low = 0
        high = seq_len

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_qk_d[None, :])
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)

    mask_n = offs_n < seq_len
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_qk_d[None, :])
    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    max_ids = seq_len
    pos_offs_n = offs_n
    # loop over rows
    for start_m in tl.range(low, high, BLOCK_M):
        bias_block_ptr = None
        if HAS_BIAS:
            bias_block_ptr = tl.make_block_ptr(
                base=bias,
                shape=(MAX_SEQ_LEN, MAX_SEQ_LEN),
                strides=(MAX_SEQ_LEN, 1),
                offsets=(start_m, start_n),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs=q_ptrs,
            dq_ptrs=dq_ptrs,
            mask_n=mask_n,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            max_ids=max_ids,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_BIAS=HAS_BIAS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            bias_block_ptr=bias_block_ptr,
        )
    # write-back
    dk = dk * alpha
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q, K, V, Grad, DQ, DK, DV,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_doh: tl.constexpr,
    seq_offsets,
    alpha: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    head_dim: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    bias,
):
    off = tl.program_id(0)
    off_batch = off // head_num
    off_head = off % head_num
    off_head = off_head.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_batch).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_batch + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    # offset pointers for batch/head
    q_offset = seq_start * stride_qm + off_head * stride_qh
    k_offset = seq_start * stride_kn + off_head * stride_kh
    v_offset = seq_start * stride_vn + off_head * stride_vh
    grad_offset = seq_start * stride_dom + off_head * stride_doh
    bias_offset = off_batch * head_num * MAX_SEQ_LEN * MAX_SEQ_LEN + off_head * MAX_SEQ_LEN * MAX_SEQ_LEN
    for start_n in range(0, seq_len, BLOCK_N):
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            Q=Q + q_offset,
            K=K + k_offset,
            V=V + v_offset,
            DOut=Grad + grad_offset,
            DQ=DQ + q_offset,
            DK=DK + k_offset,
            DV=DV + v_offset,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_qm,
            stride_dkn=stride_kn,
            stride_dvn=stride_vn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_BIAS=HAS_BIAS,
            BLOCK_D_Q=head_dim,
            BLOCK_D_V=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            bias=bias + bias_offset if HAS_BIAS else bias,
        )


def triton_hstu_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    causal: bool,
    mask: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch = seq_offsets.numel() - 1
    total_seq, head_num, head_dim = q.shape
    out = torch.empty_like(v)
    BLOCK_M = BLOCK_FWD
    BLOCK_N = BLOCK_FWD
    if total_seq == 0:
        print("error")
        return out
    has_bias = bias is not None
    core_num = get_npu_properties('num_aicore')
    tasks = total_seq * head_num // BLOCK_M // 2
    grid = (core_num, 1, 1)
    _hstu_attn_fwd[grid](q, k, v, seq_offsets, out, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), out.stride(0), out.stride(1), alpha, batch, head_num, max_seq_len, head_dim,
        causal, has_bias, core_num, tasks, BLOCK_M, BLOCK_N, mask, bias,
    )
    return out


def triton_hstu_attention_bwd(
    grad: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    causal: bool,
    bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    if grad.shape[0] == 0:
        return dq, dk, dv
    batch = seq_offsets.numel() - 1
    _, head_num, head_dim = q.shape
    has_bias = bias is not None
    grid = (batch * head_num, 1,)
    _hstu_attn_bwd[grid](q, k, v, grad, dq, dk, dv,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        grad.stride(0), grad.stride(1), seq_offsets, alpha, batch, head_num, max_seq_len, head_dim,
        causal, has_bias, BLOCK_BWD, BLOCK_BWD, bias,
    )
    return dq, dk, dv


def jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, dataType):
    seq_array = np.arange(256, max_seq_len + 1, 256)
    seq_lens = np.random.choice(seq_array, size=batch_size)
    if not np.isin(max_seq_len, seq_lens):
        seq_lens[np.random.randint(0, batch_size)] = max_seq_len
    seq_offset = torch.concat((torch.zeros((1,), dtype=torch.int64), \
                               torch.cumsum(torch.from_numpy(seq_lens), axis=0))).to(torch.int64).numpy()
    max_seq_len = np.max(seq_lens)
    total_seqs = np.sum(seq_lens)
    grad = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType)
    q = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType)
    k = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType)
    v = torch.rand((int(total_seqs), num_heads, attention_dim), dtype=dataType)
    print("batch_size:", batch_size, ", max_seq_len :", max_seq_len, ", head_nums:", num_heads, ", head_dim:", attention_dim)
    print("total_seqs:", total_seqs, "\nseq_lens:", seq_lens, "\nseq_offset:", seq_offset)

    bias = torch.empty(batch_size, num_heads, max_seq_len, max_seq_len, dtype=data_type).uniform_(-1, 1)
    mask = 1 - torch.triu(torch.ones(batch_size, num_heads, max_seq_len, max_seq_len), diagonal=1).to(torch.float32)
    return grad, q, k, v, bias, mask, max_seq_len, seq_offset


def dense_to_jagged(q, dense_tensor, seq_lens):
    tensor = torch.zeros_like(q)
    offset = 0
    for batch_id, seq_len in enumerate(seq_lens):
        tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
        offset = offset + seq_len
    return tensor


def jagged_to_dense(jagged_tensor, seq_lens, head_nums, atten_dim):
    need_pad_seq = []
    offset = 0
    for _, seq_len in enumerate(seq_lens):
        src_tensor = jagged_tensor[offset: offset + seq_len, :, :].reshape(seq_len, head_nums, atten_dim)
        need_pad_seq.append(src_tensor)
        offset = offset + seq_len

    dense_tensor = torch.nn.utils.rnn.pad_sequence(need_pad_seq, batch_first=True)
    return dense_tensor


def gloden_fwd(q, k, v, mask, alpha, seq_offset, attnBias, max_seq_len, enable_mask, enableBias, dataType):
    head_nums = q.shape[1]
    head_dim = q.shape[2]
    batch_size = attnBias.shape[0]
    seq_lens = np.zeros((batch_size, )).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
    q_dens = jagged_to_dense(q, seq_lens, head_nums, head_dim).to(dataType)
    k_dens = jagged_to_dense(k, seq_lens, head_nums, head_dim).to(dataType)
    v_dens = jagged_to_dense(v, seq_lens, head_nums, head_dim).to(dataType)
    q_dens = q_dens.permute(0, 2, 1, 3)
    k_dens = k_dens.permute(0, 2, 3, 1)
    v_dens = v_dens.permute(0, 2, 1, 3)

    qk_attn = torch.matmul(q_dens, k_dens) * alpha
    qk_attn = qk_attn.to(torch.float32)
    attnBias = attnBias.to(torch.float32)
    mask = mask.to(torch.float32)
    if enableBias:
        qk_attn = qk_attn + attnBias
    silu = F.silu(qk_attn) * (1 / max_seq_len)
    if enable_mask:
        silu = silu * mask
    silu = silu.to(dataType)
    atten_output = torch.matmul(silu, v_dens)

    atten_output = atten_output.permute(0, 2, 1, 3)
    atten_output = dense_to_jagged(q, atten_output, seq_lens)
    return atten_output.to(dataType)


def test_fwd(batch_size, max_seq_len, num_heads, attention_dim, data_type):
    alpha = 1 # 0.5
    _, q, k, v, bias, mask, max_seq_len, seq_offset = jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, data_type)
    # golden 输出
    golden_output = gloden_fwd(q, k, v, mask, alpha, seq_offset, bias, max_seq_len, True, False, data_type)
    # triton 输出
    seq_offsets = torch.tensor(seq_offset, dtype=torch.int64, device=DEVICE)
    triton_output = triton_hstu_attention_fwd(
        q=q.npu(),
        k=k.npu(),
        v=v.npu(),
        seq_offsets=seq_offsets,
        max_seq_len=int(max_seq_len),
        alpha=alpha,
        causal=True,
        mask=mask.npu(),
    )
    loss = 1e-4
    if data_type == torch.float16:
        loss = 1e-3
    elif data_type == torch.bfloat16:
        loss = 1e-2
    compare_result = torch.allclose(triton_output.cpu(), golden_output, loss, loss)

    if compare_result:
        write_result = 'ACC PASS'
    else:
        write_result = 'ACC FAIL'
    print(f"compare result: {write_result}")


def golden_bwd(grad, q, k, v, bias, mask, max_seq_len, seq_offset, enable_mask, silu_scale, enable_bias, data_type):
    def jagged_to_dense_bwd(jagged_tensor, seq_lens, max_seq_len, head_num, head_dim):
        batch_size = len(seq_lens)
        dense_tensor = torch.zeros(batch_size, max_seq_len, head_num, head_dim, dtype=jagged_tensor.dtype)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            dense_tensor[batch_id, :seq_len, :, :] = jagged_tensor[offset: offset + seq_len, :, :]
            offset = offset + seq_len

        return dense_tensor

    def dense_to_jagged_bwd(jagged_tensor, dense_tensor, seq_lens):
        tensor = torch.zeros_like(jagged_tensor)

        offset = 0
        for batch_id, seq_len in enumerate(seq_lens):
            tensor[offset: offset + seq_len, :, :] = dense_tensor[batch_id, 0: seq_len, :, :]
            offset = offset + seq_len

        return tensor

    q = q.cpu()
    k = k.cpu()
    v = v.cpu()
    grad = grad.cpu()
    head_nums = grad.shape[1]
    head_dim = grad.shape[2]
    batch_size = bias.shape[0]
    seq_lens = np.zeros((batch_size,)).astype(np.int64)
    for batch_id in range(batch_size):
        seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]
    grad_dens = jagged_to_dense_bwd(grad, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    q_dens = jagged_to_dense_bwd(q, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    k_dens = jagged_to_dense_bwd(k, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    v_dens = jagged_to_dense_bwd(v, seq_lens, max_seq_len, head_nums, head_dim).to(data_type)
    actual_seq_lens = torch.from_numpy(seq_lens).reshape(batch_size, 1, 1, 1).to(data_type)
    actual_seq_lens = torch.broadcast_to(actual_seq_lens, bias.shape)
    qk = torch.matmul(q_dens.permute(0, 2, 1, 3), k_dens.permute(0, 2, 3, 1))
    gv = torch.matmul(grad_dens.permute(0, 2, 1, 3), v_dens.permute(0, 2, 3, 1))
    qk = qk.float()
    gv = gv.float()
    bias = bias.float()
    if enable_mask:
        mask = mask.to(data_type)
        mask = mask.float()
    if enable_bias:
        bias = bias.to(data_type)
        bias = bias.float()
        qkb = qk + bias
    else:
        qkb = qk
    real_silu_scale = 1 / max_seq_len if silu_scale == 0.0 else silu_scale

    if enable_mask:
        score = F.silu(qkb) * real_silu_scale * mask
    else:
        score = F.silu(qkb) * real_silu_scale
    score = score.to(data_type)
    v_grad_dens = torch.matmul(score.permute(0, 1, 3, 2), grad_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    if enable_mask:
        bias_grad = gv * real_silu_scale * mask * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    else:
        bias_grad = gv * real_silu_scale * F.sigmoid(qkb) * (1 + qkb * (1 - F.sigmoid(qkb)))
    bias_grad = bias_grad.to(data_type)
    k_grad_dens = torch.matmul(bias_grad.permute(0, 1, 3, 2), q_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    q_grad_dens = torch.matmul(bias_grad, k_dens.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    bias_grad = bias_grad.cpu()
    q_grad_dens = q_grad_dens.cpu()
    q_grad = dense_to_jagged_bwd(q, q_grad_dens, seq_lens)
    k_grad_dens = k_grad_dens.cpu()
    k_grad = dense_to_jagged_bwd(k, k_grad_dens, seq_lens)
    v_grad_dens = v_grad_dens.cpu()
    v_grad = dense_to_jagged_bwd(v, v_grad_dens, seq_lens)
    torch.npu.synchronize()
    return q_grad, k_grad, v_grad, bias_grad


def test_bwd(batch_size, max_seq_len, num_heads, attention_dim, data_type):
    alpha = 1 # 0.5
    grad, q, k, v, bias, mask, max_seq_len, seq_offset = jagged_data_gen(batch_size, max_seq_len, num_heads, attention_dim, data_type)
    # golden 输出
    q_grad_golden, k_grad_golden, v_grad_golden, _ = golden_bwd(grad, q, k, v, bias, mask,
                                max_seq_len, seq_offset, True, 0, False, data_type)

    # triton 输出
    seq_offsets = torch.tensor(seq_offset, dtype=torch.int64, device=DEVICE)
    dq, dk, dv = triton_hstu_attention_bwd(
        grad=grad.npu(),
        q=q.npu(),
        k=k.npu(),
        v=v.npu(),
        seq_offsets=seq_offsets,
        max_seq_len=int(max_seq_len),
        alpha=alpha,
        causal=True,
    )
    loss = 1e-4
    if data_type == torch.float16:
        loss = 1e-3
    elif data_type == torch.bfloat16:
        loss = 1e-2
    q_res = torch.allclose(dq.cpu(), q_grad_golden.cpu(), loss, loss)
    k_res = torch.allclose(dk.cpu(), k_grad_golden.cpu(), loss, loss)
    v_res = torch.allclose(dv.cpu(), v_grad_golden.cpu(), loss, loss)
    if q_res and k_res and v_res:
        write_result = 'ACC PASS'
    else:
        write_result = 'ACC FAIL'
        print("dq res : ", q_res, " dk res : ", k_res, " dv res : ", v_res)
    print(f"compare result: {write_result}")


if __name__ == "__main__":
    #取值范围: 1~2048
    batch_size = 2
    #256的倍数，范围：[256,4096];
    max_seq_len = 1024
    #取值: 2/4/6/8
    num_heads = 2
    #取值: 32/64/128/256 
    attention_dim = 32
    data_type = torch.float32
    print("Running hstu attention forward test:")
    test_fwd(batch_size, max_seq_len, num_heads, attention_dim, data_type)
    print("Running hstu attention backward test:")
    test_bwd(batch_size, max_seq_len, num_heads, attention_dim, data_type)