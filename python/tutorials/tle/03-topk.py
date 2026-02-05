"""
Top-K with Triton (TLE Tutorial)
===============================

This tutorial implements Top-K over the last dimension of an (M, N) tensor and
compares against torch.topk. The shapes default to common MoE gating usage:
- input shape: (num_tokens, num_experts)
- output indices/values: (num_tokens, topk)
"""

import argparse
import os
import sys
import time
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEBUG_TIMING = os.getenv("TOPK_DEBUG_TIMING", "0") == "1"


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    mask = tl.where((x & tm) != 0, fm, tm)
    return x ^ mask


@triton.jit
def fpval_to_key_with_nan(x, x_bits):
    tm, fm = get_topmask_and_fullmask(x_bits)
    mask = tl.where((x_bits & tm) != 0, fm, tm)
    key = x_bits ^ mask
    return tl.where(x == x, key, fm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    mask = tl.where((x & tm) != 0, tm, fm)
    return x ^ mask


@triton.jit
def indx_to_key(indx, N_PAD: tl.constexpr):
    return N_PAD - indx


@triton.jit
def key_to_indx(indx, N_PAD: tl.constexpr):
    return N_PAD - indx


@triton.jit
def _topk_kernel_radix_select_impl(X, Yv, Yi,  # inputs / outputs
                                   stride_xm, stride_ym,  # strides
                                   n_rows, n_cols,  # shape
                                   K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, RADIX_BITS: tl.constexpr):
    pid = tl.program_id(0)
    row_ids = tl.arange(0, BLOCK_M)
    offs_m = pid * BLOCK_M + row_ids
    mask_m = offs_m < n_rows

    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_utype = tl.dtype(f"uint{x_nbits}")
    x_ultype = tl.dtype(f"uint{y_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    tl.static_assert(RADIX_BITS == 2, "radix select kernel supports RADIX_BITS=2 only")

    desired = tl.zeros([BLOCK_M], dtype=x_utype)
    desired_mask = tl.zeros([BLOCK_M], dtype=x_utype)
    k_to_find = tl.full([BLOCK_M], K, dtype=tl.int32)

    for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
        count0 = tl.zeros([BLOCK_M], dtype=tl.int32)
        count1 = tl.zeros([BLOCK_M], dtype=tl.int32)
        count2 = tl.zeros([BLOCK_M], dtype=tl.int32)
        count3 = tl.zeros([BLOCK_M], dtype=tl.int32)
        for t in tl.static_range(N_TILES):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
            x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key_with_nan(x, x_bits)

            matches = (x_key & desired_mask[:, None]) == desired[:, None]
            digit = (x_key >> digit_pos) & RADIX_MASK
            valid = mask_m[:, None] & mask_n[None, :] & matches
            count0 += tl.sum((valid & (digit == 0)).to(tl.int32), axis=1)
            count1 += tl.sum((valid & (digit == 1)).to(tl.int32), axis=1)
            count2 += tl.sum((valid & (digit == 2)).to(tl.int32), axis=1)
            count3 += tl.sum((valid & (digit == 3)).to(tl.int32), axis=1)

        found = tl.zeros([BLOCK_M], dtype=tl.int32)
        count = count3
        take = (found == 0) & (count >= k_to_find)
        desired = tl.where(take, desired | (tl.full([BLOCK_M], 3, dtype=x_utype) << digit_pos), desired)
        desired_mask = tl.where(
            take,
            desired_mask | (tl.full([BLOCK_M], RADIX_MASK, dtype=x_utype) << digit_pos),
            desired_mask,
        )
        k_to_find = tl.where((found == 0) & (take == 0), k_to_find - count, k_to_find)
        found = tl.where(take, 1, found)

        count = count2
        take = (found == 0) & (count >= k_to_find)
        desired = tl.where(take, desired | (tl.full([BLOCK_M], 2, dtype=x_utype) << digit_pos), desired)
        desired_mask = tl.where(
            take,
            desired_mask | (tl.full([BLOCK_M], RADIX_MASK, dtype=x_utype) << digit_pos),
            desired_mask,
        )
        k_to_find = tl.where((found == 0) & (take == 0), k_to_find - count, k_to_find)
        found = tl.where(take, 1, found)

        count = count1
        take = (found == 0) & (count >= k_to_find)
        desired = tl.where(take, desired | (tl.full([BLOCK_M], 1, dtype=x_utype) << digit_pos), desired)
        desired_mask = tl.where(
            take,
            desired_mask | (tl.full([BLOCK_M], RADIX_MASK, dtype=x_utype) << digit_pos),
            desired_mask,
        )
        k_to_find = tl.where((found == 0) & (take == 0), k_to_find - count, k_to_find)
        found = tl.where(take, 1, found)

        count = count0
        take = (found == 0) & (count >= k_to_find)
        desired = tl.where(take, desired | (tl.full([BLOCK_M], 0, dtype=x_utype) << digit_pos), desired)
        desired_mask = tl.where(
            take,
            desired_mask | (tl.full([BLOCK_M], RADIX_MASK, dtype=x_utype) << digit_pos),
            desired_mask,
        )
        k_to_find = tl.where((found == 0) & (take == 0), k_to_find - count, k_to_find)
        found = tl.where(take, 1, found)

    thr_key = desired

    min_val = tl.full([BLOCK_M, 1], float("-inf"), tl.float32).to(x_dtype)
    min_bits = min_val.to(x_utype, bitcast=True)
    min_key = fpval_to_key_with_nan(min_val, min_bits)
    min_packed = (min_key.to(x_ultype) << 16)
    global_packed = tl.broadcast_to(min_packed, (BLOCK_M, K_PAD))
    min_packed_tile = tl.broadcast_to(min_packed, (BLOCK_M, BLOCK_N))

    for t in tl.static_range(N_TILES):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
        x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))
        x_bits = x.to(x_utype, bitcast=True)
        x_key = fpval_to_key_with_nan(x, x_bits)
        idx_key = indx_to_key(offs_n, N_PAD).to(x_ultype)
        packed = (x_key.to(x_ultype) << 16) | idx_key[None, :]
        keep = (x_key >= thr_key[:, None]) & mask_m[:, None] & mask_n[None, :]
        packed = tl.where(keep, packed, min_packed_tile)
        tile_topk = tl.topk(packed, K_PAD, dim=1)
        merged = tl.join(global_packed, tile_topk)
        merged = tl.reshape(merged, BLOCK_M, 2 * K_PAD, can_reorder=False)
        global_packed = tl.topk(merged, K_PAD, dim=1)

    topk = tl.sort(global_packed, dim=1, descending=True)
    idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
    idx_raw = (topk & idx_mask).to(tl.uint32)
    y_indices = key_to_indx(idx_raw, N_PAD).to(tl.int32)
    y_values_raw = (topk >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_k[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_k[None, :]
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(Yv_ptrs, y_values, mask=store_mask)
    tl.store(Yi_ptrs, y_indices, mask=store_mask)


@triton.jit
def topk_kernel_radix_select_hist(X, Yv, Yi,  # inputs / outputs
                                  stride_xm, stride_ym,  # strides
                                  n_rows, n_cols,  # shape
                                  K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                                  BLOCK_N: tl.constexpr, RADIX_BITS: tl.constexpr):
    pid = tl.program_id(0)

    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_utype = tl.dtype(f"uint{x_nbits}")
    x_ultype = tl.dtype(f"uint{y_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1

    desired = tl.zeros([BLOCK_N], dtype=x_utype)
    desired_mask = tl.zeros([BLOCK_N], dtype=x_utype)
    k_to_find = tl.full([RADIX_SIZE], K, dtype=tl.int32)

    for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
        counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)
        bin_ids = tl.arange(0, RADIX_SIZE)
        for t in tl.static_range(N_TILES):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            X_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(X_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key_with_nan(x, x_bits)

            matches = (x_key & desired_mask) == desired
            digit = (x_key >> digit_pos) & RADIX_MASK
            valid = mask_n & matches
            for b in tl.static_range(RADIX_SIZE):
                bin_counts = tl.sum((valid & (digit == b)).to(tl.int32), axis=0)
                counts += tl.where(bin_ids == b, bin_counts, 0)

        bins = tl.arange(0, RADIX_SIZE)
        cumsum_desc = tl.cumsum(counts, reverse=True)
        cond = cumsum_desc >= k_to_find
        selected = tl.max(tl.where(cond, bins, 0), axis=0)

        is_sel = bins == selected
        count_sel = tl.sum(tl.where(is_sel, counts, 0), axis=0)
        cum_sel = tl.sum(tl.where(is_sel, cumsum_desc, 0), axis=0)
        counts_gt = cum_sel - count_sel

        selected_vec = tl.full([BLOCK_N], selected, dtype=x_utype)
        desired = desired | (selected_vec << digit_pos)
        desired_mask = desired_mask | (tl.full([BLOCK_N], RADIX_MASK, dtype=x_utype) << digit_pos)
        k_to_find = k_to_find - counts_gt

    thr_key = desired

    min_val = tl.full([1], float("-inf"), tl.float32).to(x_dtype)
    min_bits = min_val.to(x_utype, bitcast=True)
    min_key = fpval_to_key_with_nan(min_val, min_bits)
    min_packed = (min_key.to(x_ultype) << 16)
    global_packed = tl.broadcast_to(min_packed, (K_PAD, ))

    for t in tl.static_range(N_TILES):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        X_ptrs = X + pid * stride_xm + offs_n
        x = tl.load(X_ptrs, mask=mask_n, other=float("-inf"))
        x_bits = x.to(x_utype, bitcast=True)
        x_key = fpval_to_key_with_nan(x, x_bits)
        idx_key = indx_to_key(offs_n, N_PAD).to(x_ultype)
        packed = (x_key.to(x_ultype) << 16) | idx_key
        keep = (x_key >= thr_key) & mask_n
        packed = tl.where(keep, packed, tl.broadcast_to(min_packed, (BLOCK_N, )))
        tile_topk = tl.topk(packed, K_PAD, dim=0)
        merged = tl.join(global_packed, tile_topk)
        merged = tl.reshape(merged, 2 * K_PAD, can_reorder=False)
        global_packed = tl.topk(merged, K_PAD, dim=0)

    topk = tl.sort(global_packed, dim=0, descending=True)
    idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
    idx_raw = (topk & idx_mask).to(tl.uint32)
    y_indices = key_to_indx(idx_raw, N_PAD).to(tl.int32)
    y_values_raw = (topk >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K
    Yv_ptrs = Yv + pid * stride_ym + offs_k
    Yi_ptrs = Yi + pid * stride_ym + offs_k
    tl.store(Yv_ptrs, y_values, mask=mask_k)
    tl.store(Yi_ptrs, y_indices, mask=mask_k)


@triton.jit
def topk_kernel_radix_select(X, Yv, Yi,  # inputs / outputs
                             stride_xm, stride_ym,  # strides
                             n_rows, n_cols,  # shape
                             K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    _ = BLOCK_M
    topk_kernel_radix_select_hist(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_rows,
        n_cols,
        K=K,
        K_PAD=K_PAD,
        N_PAD=N_PAD,
        N_TILES=N_TILES,
        BLOCK_N=BLOCK_N,
        RADIX_BITS=4,
    )


@triton.jit
def topk_kernel(X, Yv, Yi,  # inputs / outputs
                stride_xm, stride_ym,  # strides
                n_rows, n_cols,  # shape
                K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_ids = tl.arange(0, BLOCK_M)
    offs_m = pid * BLOCK_M + row_ids
    mask_m = offs_m < n_rows

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols

    X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
    x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))

    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_utype = tl.dtype(f"uint{x_nbits}")
    x_ultype = tl.dtype(f"uint{y_nbits}")
    x_dtype = X.dtype.element_ty

    x_bits = x.to(x_utype, bitcast=True)
    x_key = fpval_to_key(x_bits)
    idx_key = indx_to_key(offs_n, BLOCK_N).to(x_ultype)
    packed = (x_key.to(x_ultype) << 16) | idx_key[None, :]

    topk = tl.topk(packed, K_PAD, dim=1)
    topk = tl.sort(topk, dim=1, descending=True)

    idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
    idx_raw = (topk & idx_mask).to(tl.uint32)
    y_indices = key_to_indx(idx_raw, BLOCK_N).to(tl.int32)

    y_values_raw = (topk >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_k[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_k[None, :]
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(Yv_ptrs, y_values, mask=store_mask)
    tl.store(Yi_ptrs, y_indices, mask=store_mask)


@triton.jit
def topk_kernel_iterative(X, Yv, Yi,  # inputs / outputs
                          stride_xm, stride_ym,  # strides
                          n_rows, n_cols,  # shape
                          K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_rows

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
    x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))

    idxs = offs_n[None, :]
    for k in tl.static_range(K):
        max_val = tl.max(x, axis=1)
        max_idx = tl.where(x == max_val[:, None], idxs, -1)
        max_idx = tl.max(max_idx, axis=1).to(tl.int32)

        Yv_ptrs = Yv + offs_m * stride_ym + k
        Yi_ptrs = Yi + offs_m * stride_ym + k
        tl.store(Yv_ptrs, max_val, mask=mask_m)
        tl.store(Yi_ptrs, max_idx, mask=mask_m)

        sel = idxs == max_idx[:, None]
        x = tl.where(sel, float("-inf"), x)


@triton.jit
def _topk_kernel_shared_radix_impl(X, Yv, Yi,  # inputs / outputs
                                   stride_xm, stride_ym,  # strides
                                   n_rows, n_cols,  # shape
                                   K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, RADIX_BITS: tl.constexpr,
                                   NUM_WARPS: tl.constexpr, USE_SMEM: tl.constexpr, HIST_BINS: tl.constexpr):
    pid = tl.program_id(0)
    row_ids = tl.arange(0, BLOCK_M)
    offs_m = pid * BLOCK_M + row_ids
    mask_m = offs_m < n_rows

    x_dtype = X.dtype.element_ty

    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_utype = tl.dtype(f"uint{x_nbits}")
    x_ultype = tl.dtype(f"uint{y_nbits}")

    BINS: tl.constexpr = 1 << RADIX_BITS
    SHIFT: tl.constexpr = x_nbits - RADIX_BITS

    bin_ids = tl.arange(0, HIST_BINS)[None, :]
    hist_counts = tl.zeros([BLOCK_M, HIST_BINS], dtype=tl.int32)

    init_vals = tl.full([BLOCK_M, K_PAD], float("-inf"), tl.float32)
    init_idx = tl.full([BLOCK_M, K_PAD], 0, tl.int32)
    init_bits = init_vals.to(x_dtype).to(x_utype, bitcast=True)
    init_key = fpval_to_key(init_bits)
    init_idx_key = indx_to_key(init_idx, N_PAD).to(x_ultype)
    global_packed = (init_key.to(x_ultype) << 16) | init_idx_key

    if HIST_BINS < BLOCK_N:
        for t in tl.static_range(N_TILES):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
            x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key(x_bits)
            bin_idx = (x_key >> SHIFT).to(tl.int32)
            tile_mask = mask_m[:, None] & mask_n[None, :]
            for b in tl.static_range(HIST_BINS):
                bin_counts = tl.sum((bin_idx == b) & tile_mask, axis=1).to(tl.int32)
                hist_counts += tl.where(bin_ids == b, bin_counts[:, None], 0)
    else:
        tl.static_assert(BLOCK_M % NUM_WARPS == 0, "BLOCK_M must be divisible by NUM_WARPS")
        ROWS_PER_WARP: tl.constexpr = BLOCK_M // NUM_WARPS
        WARP_HIST_ROWS: tl.constexpr = NUM_WARPS * BLOCK_M
        smem_warp_hist = tle.alloc(
            [WARP_HIST_ROWS, HIST_BINS],
            dtype=tl.int32,
            layout=None,
            scope=tle.smem,
            nv_mma_shared_layout=False,
        )
        warp_hist_rows = tl.arange(0, WARP_HIST_ROWS)[:, None]
        warp_hist_cols = tl.arange(0, HIST_BINS)[None, :]
        warp_hist_rows = tl.broadcast_to(warp_hist_rows, (WARP_HIST_ROWS, HIST_BINS))
        warp_hist_cols = tl.broadcast_to(warp_hist_cols, (WARP_HIST_ROWS, HIST_BINS))
        warp_hist_ptrs = tle.local_ptr(smem_warp_hist, (warp_hist_rows, warp_hist_cols))
        tl.store(warp_hist_ptrs, tl.zeros([WARP_HIST_ROWS, HIST_BINS], dtype=tl.int32))

        one = tl.full([BLOCK_M, BLOCK_N], 1, tl.int32)

        if USE_SMEM and N_TILES == 1:
            smem_tile = tle.alloc([BLOCK_M, BLOCK_N], dtype=x_dtype, layout=None, scope=tle.smem,
                                  nv_mma_shared_layout=False)
            tile_rows = tl.arange(0, BLOCK_M)[:, None]
            tile_cols = tl.arange(0, BLOCK_N)[None, :]
            tile_rows = tl.broadcast_to(tile_rows, (BLOCK_M, BLOCK_N))
            tile_cols = tl.broadcast_to(tile_cols, (BLOCK_M, BLOCK_N))
            smem_ptrs = tle.local_ptr(smem_tile, (tile_rows, tile_cols))

            offs_n = tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
            tile_mask = mask_m[:, None] & mask_n[None, :]
            tle.copy(X_ptrs, smem_tile, [BLOCK_M, BLOCK_N], mask=tile_mask, other=float("-inf"))
            x = tl.load(smem_ptrs, mask=tile_mask, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key(x_bits)
            bin_idx = (x_key >> SHIFT).to(tl.int32)
            for warp in tl.static_range(NUM_WARPS):
                row_start = warp * ROWS_PER_WARP
                row_mask = (row_ids >= row_start) & (row_ids < row_start + ROWS_PER_WARP)
                hist_rows = warp * BLOCK_M + row_ids
                hist_rows = tl.broadcast_to(hist_rows[:, None], (BLOCK_M, BLOCK_N))
                hist_addrs = tle.local_ptr(
                    smem_warp_hist,
                    (hist_rows, bin_idx),
                )
                tl.atomic_add(hist_addrs, one, mask=tile_mask & row_mask[:, None], sem="relaxed", scope="cta")
        else:
            for t in tl.static_range(N_TILES):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
                x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))
                x_bits = x.to(x_utype, bitcast=True)
                x_key = fpval_to_key(x_bits)
                bin_idx = (x_key >> SHIFT).to(tl.int32)
                tile_mask = mask_m[:, None] & mask_n[None, :]
                for warp in tl.static_range(NUM_WARPS):
                    row_start = warp * ROWS_PER_WARP
                    row_mask = (row_ids >= row_start) & (row_ids < row_start + ROWS_PER_WARP)
                    hist_rows = warp * BLOCK_M + row_ids
                    hist_rows = tl.broadcast_to(hist_rows[:, None], (BLOCK_M, BLOCK_N))
                    hist_addrs = tle.local_ptr(
                        smem_warp_hist,
                        (hist_rows, bin_idx),
                    )
                    tl.atomic_add(hist_addrs, one, mask=tile_mask & row_mask[:, None], sem="relaxed", scope="cta")

        tl.debug_barrier()
        for warp in tl.static_range(NUM_WARPS):
            hist_rows = warp * BLOCK_M + row_ids
            hist_rows = tl.broadcast_to(hist_rows[:, None], (BLOCK_M, HIST_BINS))
            bin_ids_full = tl.broadcast_to(bin_ids, (BLOCK_M, HIST_BINS))
            hist_ptrs = tle.local_ptr(
                smem_warp_hist,
                (hist_rows, bin_ids_full),
            )
            vals = tl.load(hist_ptrs, mask=(bin_ids_full < BINS), other=0)
            hist_counts += vals
    bins = tl.arange(0, HIST_BINS)[None, :]
    cumsum_desc = tl.cumsum(hist_counts, axis=1, reverse=True)
    cond = cumsum_desc >= K
    threshold_bin = tl.max(tl.where(cond, bins, 0), axis=1)

    min_bits = tl.full([BLOCK_M, 1], float("-inf"), tl.float32).to(x_dtype).to(x_utype, bitcast=True)
    min_key = fpval_to_key(min_bits)
    min_packed = (min_key.to(x_ultype) << 16) + tl.zeros([BLOCK_M, BLOCK_N], x_ultype)

    if USE_SMEM and N_TILES == 1:
        idx_key = indx_to_key(offs_n, N_PAD).to(x_ultype)
        packed = (x_key.to(x_ultype) << 16) | idx_key[None, :]
        thr = tl.maximum(threshold_bin - 1, 0)
        keep = tile_mask & (bin_idx >= thr[:, None])
        packed = tl.where(keep, packed, min_packed)
        global_packed = tl.topk(packed, K_PAD, dim=1)
    else:
        for t in tl.static_range(N_TILES):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]

            x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key(x_bits)
            bin_idx = (x_key >> SHIFT).to(tl.int32)
            idx_key = indx_to_key(offs_n, N_PAD).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key[None, :]
            thr = tl.maximum(threshold_bin - 1, 0)
            tile_mask = mask_m[:, None] & mask_n[None, :]
            keep = tile_mask & (bin_idx >= thr[:, None])
            packed = tl.where(keep, packed, min_packed)
            tile_topk = tl.topk(packed, K_PAD, dim=1)

            merged = tl.join(global_packed, tile_topk)
            merged = tl.reshape(merged, BLOCK_M, 2 * K_PAD, can_reorder=False)
            global_packed = tl.topk(merged, K_PAD, dim=1)

    topk = tl.sort(global_packed, dim=1, descending=True)
    topk = tl.sort(topk, dim=1, descending=True)

    idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
    idx_raw = (topk & idx_mask).to(tl.uint32)
    y_indices = key_to_indx(idx_raw, N_PAD).to(tl.int32)
    y_values_raw = (topk >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < K
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_k[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_k[None, :]
    store_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(Yv_ptrs, y_values, mask=store_mask)
    tl.store(Yi_ptrs, y_indices, mask=store_mask)


@triton.jit
def topk_kernel_shared_radix4(X, Yv, Yi,  # inputs / outputs
                              stride_xm, stride_ym,  # strides
                              n_rows, n_cols,  # shape
                              K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_WARPS: tl.constexpr,
                              USE_SMEM: tl.constexpr):
    _topk_kernel_shared_radix_impl(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_rows,
        n_cols,
        K=K,
        K_PAD=K_PAD,
        N_PAD=N_PAD,
        N_TILES=N_TILES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RADIX_BITS=4,
        NUM_WARPS=NUM_WARPS,
        USE_SMEM=USE_SMEM,
        HIST_BINS=16,
    )


@triton.jit
def topk_kernel_shared_radix5(X, Yv, Yi,  # inputs / outputs
                              stride_xm, stride_ym,  # strides
                              n_rows, n_cols,  # shape
                              K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_WARPS: tl.constexpr,
                              USE_SMEM: tl.constexpr):
    _topk_kernel_shared_radix_impl(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_rows,
        n_cols,
        K=K,
        K_PAD=K_PAD,
        N_PAD=N_PAD,
        N_TILES=N_TILES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RADIX_BITS=5,
        NUM_WARPS=NUM_WARPS,
        USE_SMEM=USE_SMEM,
        HIST_BINS=32,
    )


@triton.jit
def topk_kernel_shared_radix6(X, Yv, Yi,  # inputs / outputs
                              stride_xm, stride_ym,  # strides
                              n_rows, n_cols,  # shape
                              K: tl.constexpr, K_PAD: tl.constexpr, N_PAD: tl.constexpr, N_TILES: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_WARPS: tl.constexpr,
                              USE_SMEM: tl.constexpr):
    _topk_kernel_shared_radix_impl(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_rows,
        n_cols,
        K=K,
        K_PAD=K_PAD,
        N_PAD=N_PAD,
        N_TILES=N_TILES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        RADIX_BITS=6,
        NUM_WARPS=NUM_WARPS,
        USE_SMEM=USE_SMEM,
        HIST_BINS=64,
    )


# Python wrapper


def triton_topk(x: torch.Tensor, k: int, block_m: int = 128, algo: str = "topk", use_smem: bool = False,
                sweep_radix_bits: bool = False, out_vals: torch.Tensor | None = None,
                out_idx: torch.Tensor | None = None):
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")
    block_n = triton.next_power_of_2(n_cols)
    if block_n > 1024:
        raise ValueError(f"N={n_cols} too large for this tutorial kernel (max 1024)")
    if k > block_n:
        raise ValueError(f"k={k} must be <= BLOCK_N={block_n}")
    k_pad = triton.next_power_of_2(k)

    if out_vals is None:
        y_vals = torch.empty((n_rows, k), device=x.device, dtype=x.dtype)
    else:
        y_vals = out_vals
        assert y_vals.shape == (n_rows, k)
        assert y_vals.dtype == x.dtype
        assert y_vals.device == x.device
    if out_idx is None:
        y_idx = torch.empty((n_rows, k), device=x.device, dtype=torch.int32)
    else:
        y_idx = out_idx
        assert y_idx.shape == (n_rows, k)
        assert y_idx.dtype == torch.int32
        assert y_idx.device == x.device

    grid = (triton.cdiv(n_rows, block_m), )
    if algo == "iter_shared_radix_smem":
        algo = "iter_shared_radix"

    if algo == "topk":
        topk_kernel[grid](
            x,
            y_vals,
            y_idx,
            x.stride(0),
            y_vals.stride(0),
            n_rows,
            n_cols,
            K=k,
            K_PAD=k_pad,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
    elif algo == "iterative":
        topk_kernel_iterative[grid](
            x,
            y_vals,
            y_idx,
            x.stride(0),
            y_vals.stride(0),
            n_rows,
            n_cols,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=2,
        )
    elif algo == "iter_shared_radix":
        if n_cols <= 128:
            if k <= 4:
                topk_kernel_iterative[grid](
                    x,
                    y_vals,
                    y_idx,
                    x.stride(0),
                    y_vals.stride(0),
                    n_rows,
                    n_cols,
                    K=k,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                topk_kernel[grid](
                    x,
                    y_vals,
                    y_idx,
                    x.stride(0),
                    y_vals.stride(0),
                    n_rows,
                    n_cols,
                    K=k,
                    K_PAD=k_pad,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=4,
                    num_stages=2,
                )
            return y_vals, y_idx
        block_m = 1
        grid = (n_rows, )
        block_n_shared = min(256, triton.next_power_of_2(n_cols))
        n_tiles = triton.cdiv(n_cols, block_n_shared)
        n_pad = triton.next_power_of_2(n_cols)
        topk_kernel_radix_select[grid](
            x,
            y_vals,
            y_idx,
            x.stride(0),
            y_vals.stride(0),
            n_rows,
            n_cols,
            K=k,
            K_PAD=k_pad,
            N_PAD=n_pad,
            N_TILES=n_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n_shared,
            num_warps=4,
            num_stages=1,
        )
    else:
        raise ValueError(f"unknown algo: {algo}")
    return y_vals, y_idx


# Correctness


def _get_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def run_correctness(m: int, n: int, k: int, dtype: torch.dtype, algo: str):
    torch.manual_seed(0)
    if DEBUG_TIMING:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        print(f"[timing] correctness start {algo} M={m} N={n} K={k}", flush=True)
    x = torch.rand((m, n), device=DEVICE, dtype=dtype)
    t_vals, _ = torch.topk(x, k, dim=1)
    y_vals, y_idx = triton_topk(x, k, algo=algo, sweep_radix_bits=False)
    torch.testing.assert_close(y_vals, t_vals, rtol=1e-3, atol=1e-3)
    gathered = x.gather(1, y_idx.to(torch.int64))
    torch.testing.assert_close(gathered, y_vals, rtol=1e-3, atol=1e-3)
    if DEBUG_TIMING:
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        print(f"[timing] correctness {algo}: {dt:.3f} ms", flush=True)
    print(f"Correctness check passed ({algo}).")


def run_moe_correctness(batch: int, seq: int, experts: int, k: int, dtype: torch.dtype, algo: str):
    num_tokens = batch * seq
    run_correctness(num_tokens, experts, k, dtype, algo)


if "--only_unit_test" in sys.argv:
    _args = argparse.Namespace(batch=8, seq=128, experts=64, K=2, dtype="float16")
    _dtype = _get_dtype(_args.dtype)
    run_moe_correctness(_args.batch, _args.seq, _args.experts, _args.K, _dtype, "topk")
    run_moe_correctness(_args.batch, _args.seq, _args.experts, _args.K, _dtype, "iterative")
    run_moe_correctness(_args.batch, _args.seq, _args.experts, _args.K, _dtype, "iter_shared_radix")
    sys.exit(0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[
            (256, 1024, 4),
            (512, 1024, 4),
            (256, 2048, 8),
        ],
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "triton_iterative", "triton_iter_shared_radix"],
        line_names=["Triton-TopK", "Triton-Iterative", "Triton-RadixSelect"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="ms",
        plot_name="tle-topk-performance",
        args={},
    ))
def benchmark(M, N, K, provider, dtype):
    bench_warmup = 1
    bench_rep = 3
    N = int(N)
    if DEBUG_TIMING:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        print(f"[timing] bench start {provider} M={M} N={N} K={K}", flush=True)
    x = torch.rand((M, N), device=DEVICE, dtype=dtype)
    y_vals = torch.empty((M, K), device=DEVICE, dtype=dtype)
    y_idx = torch.empty((M, K), device=DEVICE, dtype=torch.int32)
    if DEBUG_TIMING:
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        print(f"[timing] alloc M={M} N={N} K={K}: {dt:.3f} ms", flush=True)
    block_m = 128
    block_n = min(256, triton.next_power_of_2(N))
    k_pad = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, block_m), )
    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton_iter_shared_radix":
        block_m_radix = min(block_m, 64)
        block_n_shared = min(256, triton.next_power_of_2(N))
        n_tiles = triton.cdiv(N, block_n_shared)
        n_pad = triton.next_power_of_2(N)
        grid_radix = (triton.cdiv(M, block_m_radix), )

        def run_kernel():
            topk_kernel_radix_select[grid_radix](
                x,
                y_vals,
                y_idx,
                x.stride(0),
                y_vals.stride(0),
                M,
                N,
                K=K,
                K_PAD=k_pad,
                N_PAD=n_pad,
                N_TILES=n_tiles,
                BLOCK_M=block_m_radix,
                BLOCK_N=block_n_shared,
                num_warps=4,
                num_stages=1,
            )

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_kernel()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            print(f"[timing] prewarm radix M={M} N={N} K={K}: {dt:.3f} ms", flush=True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    elif provider == "triton_iterative":

        def run_kernel():
            topk_kernel_iterative[grid](
                x,
                y_vals,
                y_idx,
                x.stride(0),
                y_vals.stride(0),
                M,
                N,
                K=K,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_kernel()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            print(f"[timing] prewarm iterative M={M} N={N} K={K}: {dt:.3f} ms", flush=True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    else:

        def run_kernel():
            topk_kernel[grid](
                x,
                y_vals,
                y_idx,
                x.stride(0),
                y_vals.stride(0),
                M,
                N,
                K=K,
                K_PAD=k_pad,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                num_stages=2,
            )

        if DEBUG_TIMING:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_kernel()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1e3
            print(f"[timing] prewarm topk M={M} N={N} K={K}: {dt:.3f} ms", flush=True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    return ms, max_ms, min_ms


# Main


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--seq", type=int, default=1024, help="sequence length")
    parser.add_argument("--experts", type=int, default=64, help="number of experts")
    parser.add_argument("--M", type=int, default=0, help="num rows (override batch*seq)")
    parser.add_argument("--N", type=int, default=0, help="num cols (override experts)")
    parser.add_argument("--K", type=int, default=2, help="topk")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)
    moe_n = args.N if args.N > 0 else args.experts
    check_n = min(moe_n, 256)
    check_k = min(args.K, check_n)
    run_moe_correctness(args.batch, args.seq, args.experts, check_k, dtype, "topk")
    run_moe_correctness(args.batch, args.seq, args.experts, check_k, dtype, "iterative")
    run_moe_correctness(args.batch, args.seq, args.experts, check_k, dtype, "iter_shared_radix")

    benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype)


if __name__ == "__main__":
    main()
