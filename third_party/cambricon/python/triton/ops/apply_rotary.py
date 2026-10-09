from typing import Optional, Union

import torch
import triton
import triton.language as tl


@triton.jit
def apply_cross_rotary_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, cu_seqlens, token_offsets,
        # dims
        max_seqlen, head_num, head_size, rotary_dim, ro_seqlen,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        CONJUGATE: tl.constexpr, IS_TOKEN_OFFSET_TENSOR: tl.constexpr, IS_PACKED: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1) // head_num
    pid_head = tl.program_id(1) % head_num
    m_begin = pid_m * BLOCK_M
    rotary_dim_half = rotary_dim // 2

    if not IS_PACKED:
        input = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        output = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        seqlen = max_seqlen
    else:
        start_idx = tl.load(cu_seqlens + pid_bs)
        end_idx = tl.load(cu_seqlens + pid_bs + 1)
        seqlen = end_idx - start_idx
        input = input + start_idx * stride_input_seqlen + pid_head * stride_input_headnum
        output = output + start_idx * stride_out_seqlen + pid_head * stride_out_headnum

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    if not IS_TOKEN_OFFSET_TENSOR:
        rope_offsets = tl.full((BLOCK_M, 1), value=token_offsets, dtype=tl.int32)[:, None]
    else:
        rope_offsets = tl.load(token_offsets + pid_bs * seqlen + m_offsets, mask=m_offsets < seqlen,
                               other=0)  # [block_m, 1]

    # x = [x1, x2, x3, ..., xm]
    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    x_offsets = m_offsets * stride_input_seqlen + k_offsets * stride_input_headsize
    x_mask = m_offsets < seqlen
    x = tl.load(input + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

    # x1 = [0, x1, x2, x3, ..., xm-1]
    x_offsets = m_offsets * stride_input_seqlen + (k_offsets - 1) * stride_input_headsize
    x1_mask = (m_offsets < seqlen) & (k_offsets > 0)
    x1 = tl.load(input + x_offsets, mask=x1_mask, other=0).to(tl.float32)

    # x2 = [x2, x3, x4, ..., xm, 0]
    x_offsets = m_offsets * stride_input_seqlen + (k_offsets + 1) * stride_input_headsize
    x1_mask = (m_offsets < seqlen) & (k_offsets < head_size - 1)
    x2 = tl.load(input + x_offsets, mask=x1_mask, other=0.0).to(tl.float32)

    # x_cross = [-x2, x1, -x4, x3, ....]
    x_cross = tl.where(k_offsets % 2 == 0, x2, x1)
    x_rope_coef = 1 - (k_offsets + 1) % 2 * 2
    x_cross = x_cross * x_rope_coef

    cos_begin = cos_emb + rope_offsets * rotary_dim_half
    sin_begin = sin_emb + rope_offsets * rotary_dim_half
    # cs_offset = k_offsets // 2
    cs_offset = tl.arange(0, BLOCK_K // 2)[None, :]
    cs_mask = (m_offsets < seqlen) & (cs_offset < rotary_dim_half)
    cos_half = tl.load(cos_begin + cs_offset, mask=cs_mask, other=1.0).to(tl.float32)
    sin_half = tl.load(sin_begin + cs_offset, mask=cs_mask, other=1.0).to(tl.float32)
    cos_half = tl.view(cos_half, (BLOCK_M, BLOCK_K // 2, 1))
    sin_half = tl.view(sin_half, (BLOCK_M, BLOCK_K // 2, 1))
    cos = tl.broadcast_to(cos_half, [BLOCK_M, BLOCK_K // 2, 2])
    sin = tl.broadcast_to(sin_half, [BLOCK_M, BLOCK_K // 2, 2])
    cos = tl.view(cos, (BLOCK_M, BLOCK_K))
    sin = tl.view(sin, (BLOCK_M, BLOCK_K))

    if CONJUGATE:
        sin = -sin

    x = x * cos + x_cross * sin

    out_offsets = m_offsets * stride_out_seqlen + k_offsets * stride_out_headsize
    tl.store(output + out_offsets, x, mask=(m_offsets < seqlen) & (k_offsets < head_size))


@triton.jit
def apply_opt_fold_rotary_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, cu_seqlens, token_offsets,
        # dims
        max_seqlen, head_num, head_size, rotary_dim, rope_seqlen,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        CONJUGATE: tl.constexpr, IS_TOKEN_OFFSET_TENSOR: tl.constexpr, IS_PACKED: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1)
    m_begin = pid_m * BLOCK_M
    rotary_dim_half = rotary_dim // 2

    seqlen = max_seqlen
    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    k_offsets = tl.arange(0, BLOCK_K // 2)[None, :]
    in_out_k_offsets = tl.arange(0, BLOCK_K)[None, :]
    mask = (m_offsets < seqlen)

    if not IS_TOKEN_OFFSET_TENSOR:
        ro_offsets = tl.full((BLOCK_M, 1), value=token_offsets, dtype=tl.int32)[:, None]
    else:
        ro_offsets = tl.load(token_offsets + pid_bs * seqlen + m_offsets, mask=mask, other=0)

    cos_begin = cos_emb + ro_offsets * rotary_dim_half
    sin_begin = sin_emb + ro_offsets * rotary_dim_half
    cos = tl.load(cos_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_begin + k_offsets, mask=mask, other=0.0).to(tl.float32)

    for pid_head in range(head_num):
        input_ptr = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        input_offsets = m_offsets * stride_input_seqlen + in_out_k_offsets * stride_input_headsize
        x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.view(x, (BLOCK_M, 2, BLOCK_K // 2))
        x0 = x[:, 0, :]
        x1 = x[:, 1, :]

        tsin = sin
        if CONJUGATE:
            tsin = -sin

        out0 = x0 * cos - x1 * tsin
        out1 = x0 * tsin + x1 * cos

        o = tl.zeros([BLOCK_M, 2, BLOCK_K // 2], dtype=tl.float32)
        o[:, 0, :] = out0
        o[:, 1, :] = out1
        o = tl.view(o, (BLOCK_M, BLOCK_K))
        out_offsets = m_offsets * stride_out_seqlen + in_out_k_offsets * stride_out_headsize
        output_ptr = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        tl.store(output_ptr + out_offsets, o, mask=mask)


@triton.jit
def apply_fold_rotary_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, cu_seqlens, token_offsets,
        # dims
        max_seqlen, head_num, head_size, rotary_dim, rope_seqlen,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        CONJUGATE: tl.constexpr, IS_TOKEN_OFFSET_TENSOR: tl.constexpr, IS_PACKED: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1) // head_num
    pid_head = tl.program_id(1) % head_num
    m_begin = pid_m * BLOCK_M
    rotary_dim_half = rotary_dim // 2

    if not IS_PACKED:
        input = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        output = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        seqlen = max_seqlen
    else:
        start_idx = tl.load(cu_seqlens + pid_bs)
        end_idx = tl.load(cu_seqlens + pid_bs + 1)
        seqlen = end_idx - start_idx
        input = input + start_idx * stride_input_seqlen + pid_head * stride_input_headnum
        output = output + start_idx * stride_out_seqlen + pid_head * stride_out_headnum

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    if not IS_TOKEN_OFFSET_TENSOR:
        ro_offsets = tl.full((BLOCK_M, 1), value=token_offsets, dtype=tl.int32)[:, None]
    else:
        ro_offsets = tl.load(token_offsets + pid_bs * seqlen + m_offsets, mask=(m_offsets < seqlen),
                             other=0)  # [BLOCK_M, 1]

    k_offsets = tl.arange(0, BLOCK_K // 2)[None, :]
    input_offsets = m_offsets * stride_input_seqlen + k_offsets * stride_input_headsize
    x_mask = (m_offsets < seqlen) & (k_offsets < rotary_dim_half)
    x0 = tl.load(input + input_offsets, mask=x_mask, other=0.0).to(tl.float32)
    x1 = tl.load(input + rotary_dim_half + input_offsets, mask=x_mask, other=0.0).to(tl.float32)

    cos_begin = cos_emb + ro_offsets * rotary_dim_half
    sin_begin = sin_emb + ro_offsets * rotary_dim_half
    cs_mask = (m_offsets < seqlen) & (k_offsets < rotary_dim_half)
    cos = tl.load(cos_begin + k_offsets, mask=cs_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_begin + k_offsets, mask=cs_mask, other=0.0).to(tl.float32)

    if CONJUGATE:
        sin = -sin

    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos

    # if rotary_dim != head_size:
    #     hs_offsets = tl.arange(0, BLOCK_K)[None, :]
    #     pass_offsets = m_offsets * stride_input_seqlen + hs_offsets * stride_input_headsize
    #     pass_mask = (m_offsets < seqlen) & (hs_offsets >= rotary_dim) & (hs_offsets < head_size)
    #     x_pass = tl.load(input + pass_offsets, mask=pass_mask)
    #     pass_out_offsets = m_offsets * stride_out_seqlen + hs_offsets * stride_out_headsize
    #     tl.store(output + pass_out_offsets, x_pass, mask=pass_mask)

    out_offsets = m_offsets * stride_out_seqlen + k_offsets * stride_out_headsize
    tl.store(output + out_offsets, out0, mask=x_mask)
    tl.store(output + rotary_dim_half + out_offsets, out1, mask=x_mask)


@triton.jit
def apply_cross_rotary_2d_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, cu_seqlens, token_offsets,
        # dims
        max_seqlen, head_num, head_size, rotary_dim, seqlen_ro,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        CONJUGATE: tl.constexpr, IS_PACKED: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1) // head_num
    pid_head = tl.program_id(1) % head_num
    m_begin = pid_m * BLOCK_M
    rotary_dim_half = rotary_dim // 2

    if not IS_PACKED:
        input = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        output = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        seqlen = max_seqlen
    else:
        start_idx = tl.load(cu_seqlens + pid_bs)
        end_idx = tl.load(cu_seqlens + pid_bs + 1)
        seqlen = end_idx - start_idx
        input = input + start_idx * stride_input_seqlen + pid_head * stride_input_headnum
        output = output + start_idx * stride_out_seqlen + pid_head * stride_out_headnum

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]  # [BLOCK_M, 1]

    ro_offsets0 = tl.load(token_offsets + pid_bs * 2 * seqlen + m_offsets, mask=m_offsets < seqlen, other=0)
    ro_offsets1 = tl.load(token_offsets + (pid_bs * 2 + 1) * seqlen + m_offsets, mask=m_offsets < seqlen, other=0)

    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    k_cross_offsets = k_offsets + (k_offsets + 1) % 2 * 2 - 1
    k_cross_coef = 1 - (k_offsets + 1) % 2 * 2

    #cs_offsets = tl.arange(0, BLOCK_K)[None, :] // 2
    cs_offsets = tl.arange(0, BLOCK_K // 2)[None, :]
    cs_mask0 = (m_offsets < seqlen) & (k_offsets < rotary_dim)
    cs_mask1 = (m_offsets < seqlen) & (k_offsets >= rotary_dim)
    cos_begin0 = cos_emb + ro_offsets0 * rotary_dim_half
    sin_begin0 = sin_emb + ro_offsets0 * rotary_dim_half
    cos0_half = tl.load(cos_begin0 + cs_offsets).to(tl.float32)
    sin0_half = tl.load(sin_begin0 + cs_offsets).to(tl.float32)
    cos0_half = tl.view(cos0_half, (BLOCK_M, BLOCK_K // 2, 1))
    sin0_half = tl.view(sin0_half, (BLOCK_M, BLOCK_K // 2, 1))
    cos0 = tl.broadcast_to(cos0_half, [BLOCK_M, BLOCK_K // 2, 2])
    sin0 = tl.broadcast_to(sin0_half, [BLOCK_M, BLOCK_K // 2, 2])
    cos0 = tl.view(cos0, (BLOCK_M, BLOCK_K))
    sin0 = tl.view(sin0, (BLOCK_M, BLOCK_K))
    cos0 = cos0 * cs_mask0
    sin0 = sin0 * cs_mask0

    cos_begin1 = cos_emb + ro_offsets1 * rotary_dim_half
    sin_begin1 = sin_emb + ro_offsets1 * rotary_dim_half

    cs_offsets = tl.arange(0, BLOCK_K)[None, :] // 2
    cs_mask0 = (m_offsets < seqlen) & (k_offsets < rotary_dim)
    cs_mask1 = (m_offsets < seqlen) & (k_offsets >= rotary_dim)
    cos1 = tl.load(cos_begin1 + cs_offsets - rotary_dim_half, mask=cs_mask1, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_begin1 + cs_offsets - rotary_dim_half, mask=cs_mask1, other=0.0).to(tl.float32)

    cos = cos0 + cos1
    sin = sin0 + sin1

    if CONJUGATE:
        sin = -sin

    x_offsets = m_offsets * stride_input_seqlen + k_offsets * stride_input_headsize
    x_mask = (m_offsets < seqlen) & (k_offsets < head_size)
    x = tl.load(input + x_offsets, mask=x_mask, other=0.0).to(tl.float32)

    x_offsets = m_offsets * stride_input_seqlen + (k_offsets - 1) * stride_input_headsize
    x1_mask = (m_offsets < seqlen) & (k_offsets > 0)
    x1 = tl.load(input + x_offsets, mask=x1_mask, other=0.0).to(tl.float32)

    x_offsets = m_offsets * stride_input_seqlen + (k_offsets + 1) * stride_input_headsize
    x1_mask = (m_offsets < seqlen) & (k_offsets < head_size - 1)
    x2 = tl.load(input + x_offsets, mask=x1_mask, other=0.0).to(tl.float32)

    # x_cross = [-x2, x1, -x4, x3, ....]
    x_cross = tl.where(k_offsets % 2 == 0, x2, x1)
    x_rope_coef = 1 - (k_offsets + 1) % 2 * 2
    x_cross = x_cross * x_rope_coef

    x = x_cross * sin + x * cos
    out_offsets = m_offsets * stride_out_seqlen + k_offsets * stride_input_headsize
    tl.store(output + out_offsets, x, mask=x_mask)


@triton.jit
def apply_fold_rotary_2d_kernel(
        # data pointers
        output, input, cos_emb, sin_emb, cu_seqlens, token_offsets,
        # dims
        max_seqlen, head_num, rotary_dim: tl.constexpr, seqlen_ro,
        # strides
        stride_out_batch, stride_out_seqlen, stride_out_headnum, stride_out_headsize, stride_input_batch,
        stride_input_seqlen, stride_input_headnum, stride_input_headsize,
        # meta parameters
        CONJUGATE: tl.constexpr, IS_PACKED: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bs = tl.program_id(1) // head_num
    pid_head = tl.program_id(1) % head_num
    m_begin = pid_m * BLOCK_M
    rotary_dim_half = rotary_dim // 2

    if not IS_PACKED:
        input = input + pid_bs * stride_input_batch + pid_head * stride_input_headnum
        output = output + pid_bs * stride_out_batch + pid_head * stride_out_headnum
        seqlen = max_seqlen
    else:
        start_idx = tl.load(cu_seqlens + pid_bs)
        end_idx = tl.load(cu_seqlens + pid_bs + 1)
        seqlen = end_idx - start_idx
        input = input + start_idx * stride_input_seqlen + pid_head * stride_input_headnum
        output = output + start_idx * stride_out_seqlen + pid_head * stride_out_headnum

    if m_begin >= seqlen:
        return

    m_offsets = m_begin + tl.arange(0, BLOCK_M)[:, None]
    ro_offsets0 = tl.load(token_offsets + pid_bs * 2 * seqlen + m_offsets, mask=m_offsets < seqlen, other=0)
    ro_offsets1 = tl.load(token_offsets + (pid_bs * 2 + 1) * seqlen + m_offsets, mask=m_offsets < seqlen, other=0)

    k_offsets = tl.arange(0, BLOCK_K)[None, :]
    # shape [1, block/2] [0,1,2,3,4....]
    cs_offsets = tl.arange(0, rotary_dim // 2)[None, :]
    #cs_offsets = k_offsets % rotary_dim_half
    cos0_begin = cos_emb + ro_offsets0 * rotary_dim_half
    sin0_begin = sin_emb + ro_offsets0 * rotary_dim_half
    cos0_half = tl.load(cos0_begin + cs_offsets).to(tl.float32)
    sin0_half = tl.load(sin0_begin + cs_offsets).to(tl.float32)
    cos0_half = tl.view(cos0_half, (BLOCK_M, 1, rotary_dim // 2))
    sin0_half = tl.view(sin0_half, (BLOCK_M, 1, rotary_dim // 2))
    cos0 = tl.broadcast_to(cos0_half, [BLOCK_M, 2 * BLOCK_K // rotary_dim, rotary_dim // 2])
    sin0 = tl.broadcast_to(sin0_half, [BLOCK_M, 2 * BLOCK_K // rotary_dim, rotary_dim // 2])
    cos0 = tl.view(cos0, (BLOCK_M, BLOCK_K))
    sin0 = tl.view(sin0, (BLOCK_M, BLOCK_K))
    total_mask0 = (m_offsets < seqlen) & (k_offsets < rotary_dim)
    cos0 = cos0 * total_mask0
    sin0 = sin0 * total_mask0

    cos1_begin = cos_emb + ro_offsets1 * rotary_dim_half
    sin1_begin = sin_emb + ro_offsets1 * rotary_dim_half
    cos1_half = tl.load(cos1_begin + cs_offsets).to(tl.float32)
    sin1_half = tl.load(sin1_begin + cs_offsets).to(tl.float32)
    cos1_half = tl.view(cos1_half, (BLOCK_M, 1, rotary_dim // 2))
    sin1_half = tl.view(sin1_half, (BLOCK_M, 1, rotary_dim // 2))
    cos1 = tl.broadcast_to(cos1_half, [BLOCK_M, 2 * BLOCK_K // rotary_dim, rotary_dim // 2])
    sin1 = tl.broadcast_to(sin1_half, [BLOCK_M, 2 * BLOCK_K // rotary_dim, rotary_dim // 2])
    cos1 = tl.view(cos1, (BLOCK_M, BLOCK_K))
    sin1 = tl.view(sin1, (BLOCK_M, BLOCK_K))
    total_mask1 = (m_offsets < seqlen) & (k_offsets >= rotary_dim)
    cos1 = cos1 * total_mask1
    sin1 = sin1 * total_mask1

    cos = cos0 + cos1
    sin = sin0 + sin1

    if CONJUGATE:
        sin = -sin

    k_offsets1 = k_offsets % rotary_dim
    x_offsets = m_offsets * stride_input_seqlen + k_offsets * stride_input_headsize
    x_mask = (m_offsets < seqlen)
    x_mask0 = (m_offsets < seqlen) & (k_offsets1 < rotary_dim_half)
    x_mask1 = (m_offsets < seqlen) & (k_offsets1 >= rotary_dim_half)
    x = tl.load(input + x_offsets, mask=x_mask, other=0.0).to(tl.float32)
    x0 = tl.load(input + x_offsets + rotary_dim_half, mask=x_mask0, other=0.0).to(tl.float32)
    x1 = tl.load(input + x_offsets - rotary_dim_half, mask=x_mask1, other=0.0).to(tl.float32)
    x_fold = x1 - x0

    x = x_fold * sin + x * cos
    out_offsets = m_offsets * stride_out_seqlen + k_offsets * stride_out_headsize
    tl.store(output + out_offsets, x, mask=x_mask)


def apply_cross_rotary(
    output: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    BLOCK_M: int,
    token_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate=False,
) -> torch.Tensor:
    is_packed = cu_seqlens is not None
    is_paded = not is_packed

    if is_paded:
        assert x is not None
        bs, seqlen, head_num, head_size = x.shape
    elif is_packed:
        assert max_seqlen is not None, "max_seqlen must be given for pack mode"
        total_seqlen, head_num, head_size = x.shape
        bs = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype
    ro_seqlen, rotary_dim_half = cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim <= head_size
    assert head_size in [16, 32, 64, 128, 256], "head_size is not supported"
    assert ro_seqlen >= seqlen

    if isinstance(token_offsets, torch.Tensor):
        assert token_offsets.dtype in [torch.int32, torch.int64]
        assert token_offsets.shape == (bs, seqlen)
    else:
        assert token_offsets < ro_seqlen

    num_bs_head = bs * head_num
    BLOCK_K = triton.next_power_of_2(head_size)
    grid = lambda META: ((seqlen + BLOCK_M - 1) // BLOCK_M, num_bs_head)
    apply_cross_rotary_kernel[grid](
        # data ptr
        output, x, cos, sin, cu_seqlens, token_offsets,
        # dims
        seqlen, head_num, head_size, rotary_dim, ro_seqlen,
        # strides
        output.stride(0) if is_paded else 0, output.stride(-3), output.stride(-2), output.stride(-1),
        x.stride(0) if is_paded else 0, x.stride(-3), x.stride(-2), x.stride(-1),
        # meta parameters
        conjugate, isinstance(token_offsets, torch.Tensor), is_packed, BLOCK_K, BLOCK_M)

    return output


def apply_fold_rotary(
    output: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    BLOCK_M: int,
    token_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate=False,
) -> torch.Tensor:
    is_packed = cu_seqlens is not None
    is_paded = not is_packed

    if is_paded:
        assert x is not None
        bs, seqlen, head_num, head_size = x.shape
    elif is_packed:
        assert max_seqlen is not None, "max_seqlen must be given for pack mode"
        total_seqlen, head_num, head_size = x.shape
        bs = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype
    ro_seqlen, rotary_dim_half = cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim <= head_size
    assert head_size in [16, 32, 64, 128, 256], "head_size is not supported"
    assert ro_seqlen >= seqlen

    if isinstance(token_offsets, torch.Tensor):
        assert token_offsets.dtype in [torch.int32, torch.int64]
        assert token_offsets.shape == (bs, seqlen)
    else:
        assert token_offsets < ro_seqlen

    if not is_packed and head_size == rotary_dim:
        BLOCK_K = head_size
        grid = lambda META: ((seqlen + BLOCK_M - 1) // BLOCK_M, bs)
        apply_opt_fold_rotary_kernel[grid](
            # data ptr
            output, x, cos, sin, cu_seqlens, token_offsets,
            # dims
            seqlen, head_num, head_size, rotary_dim, ro_seqlen,
            # strides
            output.stride(0) if is_paded else 0, output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if is_paded else 0, x.stride(-3), x.stride(-2), x.stride(-1),
            # meta parameters
            conjugate, isinstance(token_offsets, torch.Tensor), is_packed, BLOCK_K, BLOCK_M)
    else:
        num_bs_heads = bs * head_num
        BLOCK_K = triton.next_power_of_2(head_size)
        grid = lambda META: ((seqlen + BLOCK_M - 1) // BLOCK_M, num_bs_heads)
        apply_fold_rotary_kernel[grid](
            # data ptr
            output, x, cos, sin, cu_seqlens, token_offsets,
            # dims
            seqlen, head_num, head_size, rotary_dim, ro_seqlen,
            # strides
            output.stride(0) if is_paded else 0, output.stride(-3), output.stride(-2), output.stride(-1),
            x.stride(0) if is_paded else 0, x.stride(-3), x.stride(-2), x.stride(-1),
            # meta parameters
            conjugate, isinstance(token_offsets, torch.Tensor), is_packed, BLOCK_K, BLOCK_M)

    return output


def apply_cross_rotary_2d(
    output: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    BLOCK_M: int,
    token_offsets: torch.Tensor = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate=False,
) -> torch.Tensor:
    is_packed = cu_seqlens is not None
    is_paded = not is_packed

    if is_paded:
        assert x is not None
        bs, seqlen, head_num, head_size = x.shape
    elif is_packed:
        assert max_seqlen is not None, "max_seqlen must be given for pack mode"
        total_seqlen, head_num, head_size = x.shape
        bs = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype
    ro_seqlen, rotary_dim_half = cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim == head_size // 2
    assert head_size in [16, 32, 64, 128, 256], "head_size is not supported"
    assert ro_seqlen >= seqlen

    assert token_offsets.dtype in [torch.int32, torch.int64]
    assert token_offsets.dim() == 3
    assert token_offsets.shape == (bs, 2, seqlen)

    num_bs_head = bs * head_num
    BLOCK_K = triton.next_power_of_2(head_size)
    grid = lambda META: ((seqlen + BLOCK_M - 1) // BLOCK_M, num_bs_head)
    apply_cross_rotary_2d_kernel[grid](
        # data ptr
        output, x, cos, sin, cu_seqlens, token_offsets,
        # dims
        seqlen, head_num, head_size, rotary_dim, ro_seqlen,
        # strides
        output.stride(0) if is_paded else 0, output.stride(-3), output.stride(-2), output.stride(-1),
        x.stride(0) if is_paded else 0, x.stride(-3), x.stride(-2), x.stride(-1),
        # meta parameters
        conjugate, is_packed, BLOCK_K, BLOCK_M)

    return output


def apply_fold_rotary_2d(
    output: torch.Tensor,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    BLOCK_M: int,
    token_offsets: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    conjugate=False,
) -> torch.Tensor:
    is_packed = cu_seqlens is not None
    is_paded = not is_packed

    if is_paded:
        assert x is not None
        bs, seqlen, head_num, head_size = x.shape
    elif is_packed:
        assert max_seqlen is not None, "max_seqlen must be given for pack mode"
        total_seqlen, head_num, head_size = x.shape
        bs = cu_seqlens.shape[0] - 1
        seqlen = max_seqlen

    assert sin.shape == cos.shape
    assert sin.dtype == cos.dtype
    assert x.dtype == sin.dtype
    ro_seqlen, rotary_dim_half = cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim == head_size // 2
    assert head_size in [16, 32, 64, 128, 256], "head_size is not supported"
    assert ro_seqlen >= seqlen

    assert token_offsets.dtype in [torch.int32, torch.int64]
    assert token_offsets.dim() == 3
    assert token_offsets.shape == (bs, 2, seqlen)

    num_bs_head = bs * head_num
    BLOCK_K = triton.next_power_of_2(head_size)
    grid = lambda META: ((seqlen + BLOCK_M - 1) // BLOCK_M, num_bs_head)
    apply_fold_rotary_2d_kernel[grid](
        # data ptr
        output, x, cos, sin, cu_seqlens, token_offsets,
        # dims
        seqlen, head_num, rotary_dim, ro_seqlen,
        # strides
        output.stride(0) if is_paded else 0, output.stride(-3), output.stride(-2), output.stride(-1),
        x.stride(0) if is_paded else 0, x.stride(-3), x.stride(-2), x.stride(-1),
        # meta parameters
        conjugate, is_packed, BLOCK_K, BLOCK_M)

    return output


def apply_rotary(
    output: torch.Tensor,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    BLOCK_M: int,
    token_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    rotary_2d=False,
    interleaved=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        input: (batch, seqlen, num_heads, head_size) if cu_seqlen is None
            else (total_seqlen, num_heads, head_size) for pack mode.
        cos: (seqlen_ro, rotary_dim // 2)
        sin: (seqlen_ro, rotary_dim // 2)
        token_offsets: integer or integer tensor
        cu_seqlens: (batch + 1, ) or None
        max_seqlen: max sequence length
        rotary_2d:  2d rotary embedding
        interleaved: cross or fold
        conjugate: conjugated rotary embedding
    """
    if not rotary_2d:
        apply_rotary_func = apply_cross_rotary if interleaved else apply_fold_rotary
    else:
        apply_rotary_func = apply_cross_rotary_2d if interleaved else apply_fold_rotary_2d

    return apply_rotary_func(output, input, cos, sin, BLOCK_M, token_offsets, cu_seqlens, max_seqlen, conjugate)
