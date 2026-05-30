import triton
import triton.language as tl


@triton.jit
def _triton_rope(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rope_dim: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_rope_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    """
    This triton kernel applies rotary embedding on q and k.
    It supports rope_dim != head_dim scenario.
    It supports both neox style and non-neox style rope computation.

    Input tensor layout assumptions:

    q size: (num_tokens, num_q_heads, head_dim)
    q stride: (num_q_heads * head_dim, head_dim, 1)
    k size: (num_tokens, num_kv_heads, head_dim)
    k stride: (num_kv_heads * head_dim, head_dim, 1)
    cos/sin size: (num_tokens, rope_dim/2)
    cos/sin stride: (rope_dim/2, 1)

    Different compute pattern of IS_NEOX_STYLE:

    if IS_NEOX_STYLE:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if IS_NEOX_STYLE:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)
    """
    pid = tl.program_id(0).to(tl.int64)
    row_block_size = tl.num_programs(0)

    for row_idx in tl.range(pid, num_tokens, row_block_size):
        q_start_ptr = q_ptr + row_idx * q_row_stride
        k_start_ptr = k_ptr + row_idx * k_row_stride

        # ####################################################################
        # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
        # m of this program instance
        # ####################################################################
        cos_start_ptr = cos + row_idx * cos_row_stride
        sin_start_ptr = sin + row_idx * sin_row_stride

        cos_offsets = tl.arange(0, pad_rope_dim // 2)
        cos_mask = cos_offsets < (rope_dim // 2)
        cos_row = tl.load(cos_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)
        sin_row = tl.load(sin_start_ptr + cos_offsets, mask=cos_mask, other=0).to(tl.float32)

        # ####################################################################
        # Load the left and right half of q and k for the current
        # program instance (i.e. for the current token) separately
        # ####################################################################
        # left half of the head
        if IS_NEOX_STYLE:
            first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
            first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_rope_dim // 2)[None, :]
        else:
            first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + (2 * tl.arange(0, pad_rope_dim // 2)[None, :])
            first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + (2 * tl.arange(0, pad_rope_dim // 2)[None, :])

        first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_rope_dim // 2)[None, :]
                                                                   < (rope_dim // 2))
        first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_rope_dim // 2)[None, :]
                                                                   < (rope_dim // 2))
        q_tile_1 = tl.load(q_start_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
        k_tile_1 = tl.load(k_start_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)

        # right half of the head
        if IS_NEOX_STYLE:
            second_half_q_offsets = first_half_q_offsets + (rope_dim // 2)
            second_half_k_offsets = first_half_k_offsets + (rope_dim // 2)
        else:
            second_half_q_offsets = first_half_q_offsets + 1
            second_half_k_offsets = first_half_k_offsets + 1
        second_q_mask = first_q_mask
        second_k_mask = first_k_mask
        q_tile_2 = tl.load(q_start_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
        k_tile_2 = tl.load(k_start_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)

        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_start_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_start_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_start_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_start_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
