"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

Implementation is inspired by Dao-AILab
(see https://github.com/Dao-AILab/flash-attention)
"""

import torch
import triton

from .fwd_prefill import attn_fwd_mr, attn_fwd_mr_fast

from typing import Optional

import os

DEBUG = os.environ.get('FLASH_ATTENTION_TRITON_COREX_DEBUG', '0').lower() in ('1', 'true', 'yes')


def fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor], dropout_p: float, softmax_scale: float, causal: bool,
        window_size_left: int, window_size_right: int, softcap: float, return_softmax: bool):
    if out is None:
        out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)

    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype and q.dtype == v.dtype

    batch, seqlen_q, nheads_q, head_size = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    assert (nheads_q % nheads_k) == 0

    if DEBUG:
        print()
        print("flash_attn_triton.py::fwd inputs")
        print("q:", q.shape, q.stride())
        print("k:", k.shape, k.stride())
        print("v:", v.shape, v.stride())

    use_alibi = alibi_slopes is not None
    if use_alibi:
        assert alibi_slopes.dtype == torch.float32
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads_q
        stride_az, stride_ah = alibi_slopes.stride()
    else:
        stride_az, stride_ah = 0, 0

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 32. Otherwise should use memload
    # If smaller, the tile in the kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 32)

    softmax_lse = torch.empty((batch, nheads_q, seqlen_q), device=q.device, dtype=torch.float32)

    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    rng_state = torch.as_tensor([philox_seed, philox_offset])  # as_tensors uses the underlying data and doesnot cast

    use_dropout = (dropout_p > 0.0)
    FAST_BLOCK_M = 256
    FAST_BLOCK_N = 128
    use_prefill_fast_path = (not use_dropout and not return_softmax and not use_alibi and window_size_left == -1
                             and window_size_right == -1 and softcap == 0.0 and seqlen_q == seqlen_k
                             and (seqlen_q % FAST_BLOCK_M) == 0 and (seqlen_k % FAST_BLOCK_N) == 0
                             and head_size in {32, 64, 128})

    if use_prefill_fast_path:
        grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)
        attn_fwd_mr_fast[grid](
            q,
            k,
            v,
            softmax_scale,
            softmax_lse,
            out,
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(1),
            out.stride(3),
            softmax_lse.stride(0),
            softmax_lse.stride(1),
            softmax_lse.stride(2),
            HQ=nheads_q,
            HK=nheads_k,
            N_CTX=seqlen_k,
            IS_CAUSAL=causal,
            BLOCK_DMODEL=head_size,
        )
        return out, softmax_lse, None, rng_state

    # sd_mask is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only. This return holds no useful output aside from debugging.
    if use_dropout or return_softmax:
        sd_mask = torch.zeros((batch, nheads_q, seqlen_q, seqlen_k), device=q.device, dtype=torch.float32)
        dropout_mask = torch.zeros((batch, nheads_q, seqlen_q, seqlen_k), device=q.device, dtype=torch.float32)
        scores_strides = (sd_mask.stride(0), sd_mask.stride(1), sd_mask.stride(2), sd_mask.stride(3))
    else:
        sd_mask = None
        dropout_mask = None
        scores_strides = (0, 0, 0, 0)

    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), nheads_q, batch)

    # Default layout is BSHD
    attn_fwd_mr[grid](
        q,
        k,
        v,
        softmax_scale,
        softmax_lse,
        out,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        q.stride(3),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        k.stride(3),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(1),
        out.stride(3),
        stride_az,
        stride_ah,
        *scores_strides,
        softmax_lse.stride(0),
        softmax_lse.stride(1),
        softmax_lse.stride(2),
        None,
        None,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        sd_mask=sd_mask,
        dropout_mask=dropout_mask,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=causal,
        IS_VARLEN=False,
        USE_EXP2=True,
        USE_ALIBI=use_alibi,
        ENABLE_DROPOUT=dropout_p > 0.0,
        RETURN_SCORES=return_softmax,
        BLOCK_DMODEL=padded_d_model,
    )
    return out, softmax_lse, sd_mask if return_softmax else None, rng_state


def varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    return_softmax: bool,
):
    """Forward path for packed THD varlen inputs."""

    if seqused_k is not None:
        raise NotImplementedError("seqused_k is not supported yet")

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype and q.dtype == v.dtype
    assert cu_seqlens_q.dim() == 1 and cu_seqlens_k.dim() == 1
    assert cu_seqlens_q.numel() == cu_seqlens_k.numel()
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32

    batch = cu_seqlens_q.numel() - 1
    total_seqlen_q, nheads_q, head_size_og = q.shape
    _, nheads_k, _ = k.shape
    assert (nheads_q % nheads_k) == 0

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)

    if out is not None:
        assert out.shape == q.shape
        assert out.dtype == q.dtype

    head_size = q.shape[-1]
    if out is None:
        out = torch.empty_like(q)

    if DEBUG:
        print()
        print("flash_attn_triton.py::varlen_fwd inputs")
        print("q:", q.shape, q.stride())
        print("k:", k.shape, k.stride())
        print("v:", v.shape, v.stride())
        print("cu_seqlens_q:", cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k.shape)

    use_alibi = alibi_slopes is not None
    if use_alibi:
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dtype == torch.float32
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads_q
        stride_az, stride_ah = alibi_slopes.stride()
    else:
        stride_az, stride_ah = 0, 0

    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 32)

    use_dropout = dropout_p > 0.0
    if use_dropout or return_softmax:
        sd_mask = torch.zeros(
            (batch, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_mask = torch.zeros(
            (batch, nheads_q, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_strides = (
            sd_mask.stride(0),
            sd_mask.stride(1),
            sd_mask.stride(2),
            sd_mask.stride(3),
        )
    else:
        sd_mask = None
        dropout_mask = None
        scores_strides = (0, 0, 0, 0)

    softmax_lse = torch.empty((nheads_q, total_seqlen_q), device=q.device, dtype=torch.float32)
    stride_lse_z = 0
    stride_lse_h, stride_lse_m = softmax_lse.stride()

    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    rng_state = torch.as_tensor([philox_seed, philox_offset])

    grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), nheads_q, batch)
    q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
    v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
    o_strides = (0, out.stride(1), out.stride(0), out.stride(2))

    attn_fwd_mr[grid](
        q,
        k,
        v,
        softmax_scale,
        softmax_lse,
        out,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        stride_az,
        stride_ah,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        sd_mask=sd_mask,
        dropout_mask=dropout_mask,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=max_seqlen_q,
        MAX_SEQLENS_K=max_seqlen_k,
        IS_CAUSAL=causal,
        IS_VARLEN=True,
        USE_EXP2=True,
        USE_ALIBI=use_alibi,
        ENABLE_DROPOUT=dropout_p > 0.0,
        RETURN_SCORES=return_softmax,
        BLOCK_DMODEL=padded_d_model,
    )

    return out, softmax_lse, sd_mask if return_softmax else None, rng_state


def _flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cum_seq_q: Optional[torch.Tensor],
    cum_seq_k: Optional[torch.Tensor],
    max_q: int,
    max_k: int,
    dropout_p: float,
    is_causal: bool,
    return_debug_mask: bool,
    *,
    scale: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    seqused_k: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
):
    """Flash attention forward entry point aligned with aten.

    This interface is intended to mirror
    `torch.ops.aten._flash_attention_forward` while
    using this Triton implementation underneath.

    Dense inputs use the BSHD layout
    `(batch_size, seqlen, nheads, headdim)`. When both `cum_seq_q` and
    `cum_seq_k` are provided, the varlen path is enabled and packed THD tensors
    `(total_seqlen, nheads, headdim)` are expected instead.

    Arguments:
        query: Query tensor. Dense path expects shape
            `(batch_size, seqlen_q, nheads_q, headdim)`, while the varlen path
            expects `(total_seqlen_q, nheads_q, headdim)`.
        key: Key tensor. Dense path expects
            `(batch_size, seqlen_k, nheads_k, headdim)`, while the varlen path
            expects `(total_seqlen_k, nheads_k, headdim)`.
        value: Value tensor matching the layout of `key`.
        cum_seq_q: Optional cumulative sequence lengths for packed query input.
            When this is provided, `cum_seq_k` must also be provided and the
            varlen kernel path is selected.
        cum_seq_k: Optional cumulative sequence lengths for packed key/value
            input. Must be provided together with `cum_seq_q`.
        max_q: Maximum query sequence length. For dense inputs this is expected
            to match `query.shape[1]`; for varlen inputs this is the per-context
            max sequence length used to size the launch grid and debug mask.
        max_k: Maximum key/value sequence length. For dense inputs this is
            expected to match `key.shape[1]`; for varlen inputs this is the
            per-context max sequence length used to size the launch grid and
            debug mask.
        dropout_p: Dropout probability. This is forwarded to the kernel.
        is_causal: Whether to apply the right-aligned causal mask.
        return_debug_mask: Whether to return the encoded debug softmax mask.
            This is mainly intended for validation/debugging.
        scale: Scaling applied to `QK^T` before softmax. Defaults to
            `1 / sqrt(headdim)` when omitted.
        window_size_left: Left local-attention window size. `None` maps to `-1`,
            meaning no left window limit.
        window_size_right: Right local-attention window size. `None` maps to
            `-1`, meaning no right window limit.
        seqused_k: Optional per-token valid lengths for K/V. This aten-compatible
            argument is reserved for future use and is not supported yet.
        alibi_slopes: Optional ALiBi slopes. The current implementation expects
            an fp32 tensor of shape `(batch_size, nheads_q)`.

    Return:
        A 5-tuple matching the aten forward API contract:
        1. `output`: Attention output matching the input layout.
        2. `softmax_lse`: Base-2 log-sum-exp of the attention logits with shape
           `(batch_size, nheads_q, seqlen_q)` for dense inputs and
           `(nheads_q, total_seqlen_q)` for varlen inputs.
        3. `rng_state`: RNG seed tensor used by the dropout path.
        4. `unused`: Reserved tensor slot kept for aten compatibility. The
           current implementation returns the RNG offset tensor here.
        5. `debug_attn_mask`: Encoded debug softmax mask when
           `return_debug_mask=True`, otherwise `None`. For varlen inputs the
           debug mask is allocated as `(batch_size, nheads_q, max_q, max_k)`.
    """
    use_varlen = cum_seq_q is not None or cum_seq_k is not None
    if use_varlen and (cum_seq_q is None or cum_seq_k is None):
        raise ValueError("cum_seq_q and cum_seq_k must both be provided for varlen inputs")
    if seqused_k is not None:
        raise NotImplementedError("seqused_k is not supported yet")

    if scale is None:
        scale = query.shape[-1]**(-0.5)

    if window_size_left is None:
        window_size_left = -1
    if window_size_right is None:
        window_size_right = -1

    head_size_og = query.size(-1)
    if head_size_og % 32 != 0:
        pad_size = 32 - head_size_og % 32
        query = torch.nn.functional.pad(query, [0, pad_size])
        key = torch.nn.functional.pad(key, [0, pad_size])
        value = torch.nn.functional.pad(value, [0, pad_size])

    if use_varlen:
        out_padded, softmax_lse, debug_attn_mask, rng_state = varlen_fwd(
            query,
            key,
            value,
            None,
            cum_seq_q,
            cum_seq_k,
            seqused_k,
            alibi_slopes,
            max_q,
            max_k,
            dropout_p,
            scale,
            is_causal,
            window_size_left,
            window_size_right,
            0.0,
            return_debug_mask,
        )
    else:
        out_padded, softmax_lse, debug_attn_mask, rng_state = fwd(
            query,
            key,
            value,
            None,
            alibi_slopes,
            dropout_p,
            scale,
            is_causal,
            window_size_left,
            window_size_right,
            0.0,  # softcap
            return_softmax=return_debug_mask,
        )
    out = out_padded[..., :head_size_og]
    unused = rng_state[1].clone()
    return out, softmax_lse, rng_state[0].clone(), unused, debug_attn_mask
