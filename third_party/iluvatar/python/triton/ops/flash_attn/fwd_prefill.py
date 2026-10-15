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
import triton.language as tl

from typing import Literal, Optional, Union

version = tuple(map(int, triton.__version__.split('.')[:2]))
if version >= (2, 0) and version < (3, 0):
    from triton.language.math import fast_dividef
elif version >= (3, 0):
    from triton.language.extra.corex.libdevice import fast_dividef
else:
    raise ValueError(f"不支持的 Triton 版本: {triton.__version__}")


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second, stride):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, stride=stride, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, stride=stride, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, stride=stride, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs, stride=stride)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n):
    # Keep the diagonal right-aligned when seqlen_q != seqlen_k.
    relative_pos_block = offs_m[None, :] + seqlen_k - seqlen_q - offs_n[:, None]
    return -1 * alibi_slope * tl.abs(relative_pos_block)


@triton.jit
def _attn_fwd_mr_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn, stride_sn, start_m, actual_seqlen_k,
                       actual_seqlen_q, dropout_p, philox_seed, philox_ptrs, sd_mask_ptrs, dropout_mask_ptrs,
                       alibi_slope, block_min, block_max, offs_n_causal, n_extra_tokens, IS_CAUSAL: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M, OFFS_N,
                       MASK_STEPS: tl.constexpr, PADDED_HEAD: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE,
                       USE_EXP2: tl.constexpr, USE_ALIBI: tl.constexpr, RETURN_SCORES: tl.constexpr,
                       ENABLE_DROPOUT: tl.constexpr, ACCUMULATOR_TYPE):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulators
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_n, k_offs_d, actual_seqlen_k, ACTUAL_BLOCK_DMODEL, stride_kn)

        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=ACCUMULATOR_TYPE)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[:, None]
                mask = size_n < boundary_m[None, :]
                qk = tl.where(mask, qk, float("-inf"))

        # score/mask tile is [BLOCK_N, BLOCK_M]
        q_mask = OFFS_M[None, :] < actual_seqlen_q
        k_mask = (start_n + OFFS_N[:, None]) < actual_seqlen_k
        p_mask = q_mask & k_mask

        qk += tl.dot(k, q)
        qk_scaled = qk * SM_SCALE

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[None, :] >= causal_boundary[:, None]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))

        if USE_ALIBI:
            global_n_positions = start_n + OFFS_N
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, OFFS_M, global_n_positions)
            if USE_EXP2:
                qk_scaled += alibi_block * RCP_LN2
            else:
                qk_scaled += alibi_block

        v = load_fn(v_ptrs, k_offs_d, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k, stride_vn)
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 0))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[None, :]
        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)
        l_ij = tl.sum(p, 0)

        if ENABLE_DROPOUT:
            rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
            dropout_mask = rng_output > dropout_p

            if RETURN_SCORES:
                # return scores with negative values for dropped vals
                sd_mask = tl.where(dropout_mask, p, -p)
                tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            tl.store(sd_mask_ptrs, p, mask=p_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        # -- update l_i
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[None, :]
        acc += tl.dot(v, p.to(v.type.element_ty))

        # update m_i and l_i
        m_i = m_ij

        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn
    return acc, l_i, m_i


launch_configs_mr = [triton.Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=16, num_stages=1)]

# Hoisted: inspect.getblock (Python < 3.10.10, gh-102647) drops `def` when heuristics lambdas are inline.
_ATTN_FWD_MR_HEURISTICS = {
    'EVEN_M': lambda args: (not args['IS_VARLEN']) and (args['MAX_SEQLENS_Q'] % args['BLOCK_M'] == 0),
    'EVEN_N': lambda args: (not args['IS_VARLEN']) and (args['MAX_SEQLENS_K'] % args['BLOCK_N'] == 0),
}


@triton.autotune(
    configs=launch_configs_mr,
    key=["MAX_SEQLENS_Q", "MAX_SEQLENS_K"],
)
@triton.heuristics(_ATTN_FWD_MR_HEURISTICS)
@triton.jit
def attn_fwd_mr(
    Q,
    K,
    V,
    SM_SCALE,
    LSE,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    stride_az,
    stride_ah,
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    dropout_mask,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634  # 1/log(2)
        SM_SCALE *= RCP_LN2
    # set params
    ACCUMULATOR_TYPE = tl.float32

    # compute offsets
    pid_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    if IS_CAUSAL:
        num_m_blocks = tl.cdiv(MAX_SEQLENS_Q, BLOCK_M)
        start_m = tl.where(off_h_q % 2 == 0, num_m_blocks - pid_m - 1, pid_m)
    else:
        start_m = pid_m
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # Grid is based on max_seqlen_q, so some blocks must exit early.
        if start_m * BLOCK_M >= seqlen_q:
            return

        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = tl.cdiv((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.

        if n_blocks <= 0:
            o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_d[:, None] * stride_od + offs_m[None, :] * stride_om
            acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[None, :] < seqlen_q
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)

            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
            l_ptrs = l_offset + offs_m * stride_lse_m
            l = tl.full([BLOCK_M], value=0.0, dtype=ACCUMULATOR_TYPE)
            if IS_VARLEN:
                l_ptrs_mask = offs_m < seqlen_q
            else:
                l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_d[:, None] * stride_qd + offs_m[None, :] * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vn
    v_ptrs = v_offset + offs_d[:, None] * stride_vd + offs_n[None, :] * stride_vn

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = 0.0

    if RETURN_SCORES:
        sd_mask_offset = sd_mask + off_z * stride_sz + off_h_q * stride_sh  #+ cu_seqlens_q_start * stride_sm
        sd_mask_ptrs = sd_mask_offset + offs_n[:, None] * stride_sn + offs_m[None, :] * stride_sm
    else:
        sd_mask_ptrs = None

    if ENABLE_DROPOUT:
        dropout_mask_offset = dropout_mask + off_z * stride_sz + off_h_q * stride_sh  #+ cu_seqlens_q_start * stride_sm
        dropout_mask_ptrs = dropout_mask_offset + offs_n[:, None] * stride_sn + offs_m[None, :] * stride_sm
        batch_philox_offset = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh  #+ cu_seqlens_q_start * stride_sm
        philox_ptrs = batch_philox_offset + offs_n[:, None] * stride_sn + offs_m[None, :] * stride_sm
    else:
        dropout_mask_ptrs = None
        philox_ptrs = 0

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=ACCUMULATOR_TYPE) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=ACCUMULATOR_TYPE)

    if EVEN_M and not PADDED_HEAD:
        q = tl.load(q_ptrs, stride=stride_qm)
    else:
        q_ptrs_mask = offs_m[None, :] < seqlen_q
        if PADDED_HEAD:
            q_ptrs_mask = q_ptrs_mask & (offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        q = tl.load(q_ptrs, stride=stride_qm, mask=q_ptrs_mask, other=0.0)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N

    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_mr_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn, stride_sn, start_m,
                                           seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs, sd_mask_ptrs,
                                           dropout_mask_ptrs, alibi_slope, block_min, block_max, 0,  # offs_n_causal
                                           0,  # n_extra_tokens
                                           False,  # IS_CAUSAL
                                           BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n, False,  # MASK_STEPS
                                           PADDED_HEAD, ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_EXP2=USE_EXP2,
                                           USE_ALIBI=USE_ALIBI, RETURN_SCORES=RETURN_SCORES,
                                           ENABLE_DROPOUT=ENABLE_DROPOUT, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn

        if RETURN_SCORES:
            sd_mask_ptrs += n_full_blocks * BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sn
            philox_ptrs += n_full_blocks * BLOCK_N * stride_sn

        MASK_STEPS: tl.constexpr = not EVEN_N
        acc, l_i, m_i = _attn_fwd_mr_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn, stride_sn, start_m,
                                           seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs, sd_mask_ptrs,
                                           dropout_mask_ptrs, alibi_slope, block_min, block_max, offs_n_causal,
                                           n_extra_tokens, IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                           MASK_STEPS, PADDED_HEAD, ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_EXP2=USE_EXP2,
                                           USE_ALIBI=USE_ALIBI, RETURN_SCORES=RETURN_SCORES,
                                           ENABLE_DROPOUT=ENABLE_DROPOUT, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE)

    # epilogue
    acc = fast_dividef(acc, l_i[None, :])
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[None, :] >= out_mask_boundary[:, None]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + cu_seqlens_q_start * stride_lse_m
    l_ptrs = l_offset + offs_m * stride_lse_m
    if USE_EXP2:
        softmax_lse = m_i + tl.math.log2(l_i)
    else:
        softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL:
        # zero out nans caused by -infs when doing causal
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)  # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse)  # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_d[:, None] * stride_od + offs_m[None, :] * stride_om
    o_ptrs_mask = tl.full([BLOCK_DMODEL, BLOCK_M], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[None, :] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    tl.store(o_ptrs, acc, mask=o_ptrs_mask)


@triton.autotune(
    configs=launch_configs_mr,
    key=["N_CTX"],
)
@triton.jit
def attn_fwd_mr_fast(
    Q,
    K,
    V,
    SM_SCALE,
    LSE,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    SM_SCALE *= 1.44269504  # 1 / ln(2)
    pid_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    if IS_CAUSAL:
        num_m_blocks = tl.cdiv(N_CTX, BLOCK_M)
        start_m = tl.where(off_h_q % 2 == 0, num_m_blocks - pid_m - 1, pid_m)
    else:
        start_m = pid_m

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh
    q_ptrs = q_offset + offs_d[:, None] * stride_qd + offs_m[None, :] * stride_qm

    k_offset = K + off_z * stride_kz + off_h_k * stride_kh
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd

    v_offset = V + off_z * stride_vz + off_h_k * stride_vh
    v_ptrs = v_offset + offs_d[:, None] * stride_vd + offs_n[None, :] * stride_vn

    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)

    q = tl.load(q_ptrs, stride=stride_qm)

    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(k_ptrs, stride=stride_kn)
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(k, q)
        qk *= SM_SCALE
        if IS_CAUSAL:
            if start_n >= start_m * BLOCK_M:
                qk = tl.where(offs_m[None, :] >= (start_n + offs_n[:, None]), qk, float("-inf"))

        v = tl.load(v_ptrs, stride=stride_vn)
        m_curr = tl.maximum(tl.max(qk, 0), m_prev)
        alpha = tl.math.exp2(m_prev - m_curr)
        p = tl.math.exp2(qk - m_curr[None, :])
        l_prev = l_prev * alpha + tl.sum(p, 0)
        acc *= alpha[None, :]
        acc += tl.dot(v, p.to(Q.dtype.element_ty))
        m_prev = m_curr

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    acc = fast_dividef(acc, l_prev[None, :])

    l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h
    l_ptrs = l_offset + offs_m * stride_lse_m
    lse = m_prev + tl.math.log2(l_prev)
    tl.store(l_ptrs, lse)

    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh
    out_ptrs = o_offset + offs_d[:, None] * stride_od + offs_m[None, :] * stride_om
    tl.store(out_ptrs, acc)
