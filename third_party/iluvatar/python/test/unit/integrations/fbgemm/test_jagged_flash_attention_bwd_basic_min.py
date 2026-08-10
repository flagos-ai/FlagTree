import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _jagged_flash_attention_bwd_preprocess_basic_kernel(
    o_ptr,
    o_offset_ptr,
    do_ptr,
    delta_ptr,
    stride_om,
    stride_od,
    max_seq_len,
    D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    begin_o = tl.load(o_offset_ptr + pid_batch)
    end_o = tl.load(o_offset_ptr + pid_batch + 1)

    M = end_o - begin_o
    M = tl.minimum(M, max_seq_len)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_od = tl.arange(0, BLOCK_SIZE_D)

    o_offsets = (offs_om[:, None] * stride_om + offs_od[None, :] * stride_od + begin_o * stride_om)
    o_ptrs = o_ptr + o_offsets
    do_ptrs = do_ptr + o_offsets
    o_mask = (offs_om[:, None] < M) & (offs_od[None, :] < D)

    o = tl.load(o_ptrs, mask=o_mask)
    do = tl.load(do_ptrs, mask=o_mask)

    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + begin_o + offs_om, delta, mask=offs_om < M)


@triton.jit
def _jagged_flash_attention_bwd_basic_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    offset_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    do_ptr,
    delta_ptr,
    lse_ptr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    stride_dqm,
    stride_dqd,
    stride_dkn,
    stride_dkd,
    stride_dvn,
    stride_dvd,
    stride_dom,
    stride_dod,
    max_seq_len,
    D: tl.constexpr,
    use_mask: tl.constexpr,
    allow_tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(axis=1)

    begin = tl.load(offset_ptr + pid_batch)
    end = tl.load(offset_ptr + pid_batch + 1)

    M = tl.minimum(end - begin, max_seq_len)

    pid_n = tl.program_id(axis=0)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    q_ptrs = (q_ptr + begin * stride_qm + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd))

    k_ptrs = (k_ptr + begin * stride_kn + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd))

    v_ptrs = (v_ptr + begin * stride_vn + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd))

    do_ptrs = (do_ptr + begin * stride_dom + (offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod))

    k = tl.load(k_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    v = tl.load(v_ptrs, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    dv = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    for begin_m in range(0, M, BLOCK_SIZE_M):
        offs_m_temp = begin_m + offs_m

        q = tl.load(q_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        qk = tl.dot(q, tl.trans(k), allow_tf32=allow_tf32)

        mn_mask = (offs_m_temp[:, None] < M) & (offs_n[None, :] < M)

        lse_i = tl.load(lse_ptr + offs_m_temp + begin, mask=offs_m_temp < M)

        p = tl.exp(qk - lse_i[:, None])
        p = tl.where(mn_mask, p, 0.0)
        p /= max_seq_len
        p_masked = p

        attn_mask = None
        if use_mask:
            attn_mask = offs_m_temp[:, None] - offs_n[None, :]
            attn_mask = tl.where(mn_mask, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            p_masked = tl.where(attn_mask > 0, p, 0.0)

        p_masked = p_masked.to(do_ptr.dtype.element_ty)
        do = tl.load(do_ptrs, mask=((offs_d[None, :] < D) & (offs_m_temp[:, None] < M)))
        dv += tl.dot(tl.trans(p_masked), do, allow_tf32=allow_tf32)
        dp = tl.dot(do, tl.trans(v), allow_tf32=allow_tf32)

        Di = tl.load(delta_ptr + offs_m_temp + begin, mask=offs_m_temp < M)
        dp_masked = dp
        if use_mask:
            dp_masked = tl.where(attn_mask > 0, dp, 0.0)

        ds = p * (dp_masked - Di[:, None] * max_seq_len)
        ds = ds.to(q_ptr.dtype.element_ty)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=allow_tf32)

        q_ptrs += BLOCK_SIZE_M * stride_qm
        do_ptrs += BLOCK_SIZE_M * stride_dom

    dk_ptrs = (dk_ptr + begin * stride_dkn + (offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd))

    dv_ptrs = (dv_ptr + begin * stride_dvn + (offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd))

    tl.store(dk_ptrs, dk, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))
    tl.store(dv_ptrs, dv, mask=((offs_d[None, :] < D) & (offs_n[:, None] < M)))

    start_m = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_m_curr = start_m + tl.arange(0, BLOCK_SIZE_N)

    dq_ptrs_curr = (dq_ptr + begin * stride_dqm + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :] * stride_dqd))

    dq_curr = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_D], dtype=tl.float32)

    q_ptrs_curr = (q_ptr + begin * stride_qm + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd))

    q_curr = tl.load(q_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M)))

    lse_i_curr = tl.load(lse_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    do_ptrs_curr = (do_ptr + begin * stride_dom + (offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dod))

    do_curr = tl.load(do_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M)))
    Di_curr = tl.load(delta_ptr + offs_m_curr + begin, mask=offs_m_curr < M)

    block_start = 0
    while block_start < M:
        offs_n_curr = block_start + tl.arange(0, BLOCK_SIZE_M)

        k_ptrs_curr = (k_ptr + begin * stride_kn + (offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kd))
        v_ptrs_curr = (v_ptr + begin * stride_vn + (offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd))

        k_curr = tl.load(k_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M)))
        v_curr = tl.load(v_ptrs_curr, mask=((offs_d[None, :] < D) & (offs_n_curr[:, None] < M)))

        qk_curr = tl.dot(q_curr, tl.trans(k_curr), allow_tf32=allow_tf32)
        mn_mask_curr = (offs_m_curr[:, None] < M) & (offs_n_curr[None, :] < M)

        p_curr = tl.exp(qk_curr - lse_i_curr[:, None])
        p_curr = tl.where(mn_mask_curr, p_curr, 0.0)
        p_curr /= max_seq_len

        dp_curr = tl.dot(do_curr, tl.trans(v_curr), allow_tf32=allow_tf32)
        dp_curr_masked = dp_curr

        if use_mask:
            attn_mask = offs_m_curr[:, None] - offs_n_curr[None, :]
            attn_mask = tl.where(mn_mask_curr, attn_mask, 0.0)
            attn_mask = tl.where(attn_mask > 0, 0.0, 1.0)
            dp_curr_masked = tl.where(attn_mask > 0, dp_curr, 0.0)

        ds_curr = p_curr * (dp_curr_masked - Di_curr[:, None] * max_seq_len)
        ds_curr = ds_curr.to(k_ptr.dtype.element_ty)
        dq_curr += tl.dot(ds_curr, k_curr, allow_tf32=allow_tf32)
        block_start += BLOCK_SIZE_M

    tl.store(dq_ptrs_curr, dq_curr, mask=((offs_d[None, :] < D) & (offs_m_curr[:, None] < M)))


def _torch_reference(q, k, v, offsets, max_seq_len, use_mask):
    out = torch.zeros_like(q)
    B = offsets.numel() - 1
    for b in range(B):
        begin = int(offsets[b].item())
        end = int(offsets[b + 1].item())
        seqlen = min(end - begin, max_seq_len)
        if seqlen <= 0:
            continue
        q_seq = q[begin:begin + seqlen]
        k_seq = k[begin:begin + seqlen]
        v_seq = v[begin:begin + seqlen]
        score = q_seq @ k_seq.transpose(0, 1)
        p = torch.softmax(score, dim=1) / max_seq_len
        if use_mask:
            mask = torch.triu(torch.ones((seqlen, seqlen), device=q.device, dtype=torch.bool))
            p = torch.where(mask, p, torch.zeros_like(p))
        out[begin:begin + seqlen] = p @ v_seq
    return out


def _run_triton_bwd(q, k, v, offsets, dout, max_seq_len, use_mask, allow_tf32):
    block_size_m = 32
    block_size_n = 32
    block_size_d = max(triton.next_power_of_2(q.size(1)), 16)

    o = _torch_reference(q, k, v, offsets, max_seq_len, use_mask)
    delta = torch.empty((q.size(0), ), device=q.device, dtype=q.dtype)

    B = offsets.numel() - 1
    pre_grid = (triton.cdiv(max_seq_len, block_size_m), B)
    _jagged_flash_attention_bwd_preprocess_basic_kernel[pre_grid](
        o,
        offsets,
        dout,
        delta,
        o.stride(0),
        o.stride(1),
        max_seq_len,
        o.size(1),
        block_size_m,
        block_size_d,
    )

    lse = torch.empty((q.size(0), ), device=q.device, dtype=q.dtype)
    for b in range(B):
        begin = int(offsets[b].item())
        end = int(offsets[b + 1].item())
        seqlen = min(end - begin, max_seq_len)
        if seqlen <= 0:
            continue
        q_seq = q[begin:begin + seqlen]
        k_seq = k[begin:begin + seqlen]
        score = q_seq @ k_seq.transpose(0, 1)
        lse[begin:begin + seqlen] = torch.logsumexp(score, dim=1)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    grid = (triton.cdiv(max_seq_len, block_size_n), B)
    _jagged_flash_attention_bwd_basic_kernel[grid](
        q,
        k,
        v,
        o,
        offsets,
        dq,
        dk,
        dv,
        dout,
        delta,
        lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        dq.stride(0),
        dq.stride(1),
        dk.stride(0),
        dk.stride(1),
        dv.stride(0),
        dv.stride(1),
        dout.stride(0),
        dout.stride(1),
        max_seq_len,
        q.size(1),
        use_mask=use_mask,
        allow_tf32=allow_tf32,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_D=block_size_d,
    )
    return dq, dk, dv, o


def test_jagged_flash_attention_bwd_basic_min_repro():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    device = torch.device("cuda")

    B = 1
    max_seq_len = 1
    D = 17
    use_mask = False
    allow_tf32 = False

    q = torch.rand((1, D), device=device, dtype=torch.float32, requires_grad=True)
    k = torch.rand((1, D), device=device, dtype=torch.float32, requires_grad=True)
    v = torch.rand((1, D), device=device, dtype=torch.float32, requires_grad=True)
    dout = torch.rand_like(q) * 0.01
    offsets = torch.tensor([0, 1], device=device, dtype=torch.int64)
    assert offsets.numel() == B + 1

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = _torch_reference(q_ref, k_ref, v_ref, offsets, max_seq_len, use_mask)
    out_ref.backward(dout)

    dq, dk, dv, out = _run_triton_bwd(
        q.detach(),
        k.detach(),
        v.detach(),
        offsets,
        dout,
        max_seq_len,
        use_mask,
        allow_tf32,
    )

    torch.testing.assert_close(out, out_ref.detach(), atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(dv, v_ref.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(dk, k_ref.grad, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(dq, q_ref.grad, atol=1e-4, rtol=1e-3)
