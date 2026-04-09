"""
DeepSeek V3-2 Top-K Selector with Triton and TLE (TLE Tutorial)
==============================================================

This tutorial adapts the TileLang DeepSeek V3-2 top-k selector example and
implements a TLE kernel:
- A TRT-style 4-step selector in TLE:
  step0 (fp16-mapped 11-bit) + step1/2/3 (uint32-mapped 11/11/10-bit)
  with shared-memory histogram/final-sort flow.

If TileLang is installed, the script will also run the original TileLang kernel
and compare correctness and performance.

Notes
-----
- Input dtype is assumed to be float32 for the 32-bit radix refinement.
"""

# %%
# Setup
# -----

import argparse
import hashlib
import urllib.request
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

try:
    import tilelang
    import tilelang.language as T

    _HAVE_TILELANG = True
except Exception:  # pragma: no cover - optional dependency
    tilelang = None
    T = None
    _HAVE_TILELANG = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()
RADIX_BITS = 8
RADIX = 1 << RADIX_BITS
TLE_FIXED_BLOCK_SIZE = 512
TLE_FIXED_NUM_WARPS = TLE_FIXED_BLOCK_SIZE // 32
TLE_FIXED_NUM_STAGES = 1
TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD = 12288
TLE_SMEM_BLOCK_SIZE = 1024
TLE_SMEM_NUM_WARPS = TLE_SMEM_BLOCK_SIZE // 32
TLE_SMEM_NUM_STAGES = 1
TLE_SMEM_INPUT_SIZE = 4096
TLE_SMEM_CLUSTER_SIZE = 8
BLOCK_CLUSTER_MESH_8 = tle.device_mesh({"block_cluster": [("cluster_x", TLE_SMEM_CLUSTER_SIZE)]})

# %%
# Key conversions
# ---------------


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits


@triton.jit
def _convert_to_uint16_hi8(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return (bits >> 8).to(tl.int32)


@triton.jit
def _convert_to_uint16_hi11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return (bits >> 5).to(tl.int32)


@triton.jit
def _convert_to_trt_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _convert_to_trt_uint16_hi11(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
    sign_set = (bits & sign_mask) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    return (mapped >> 5).to(tl.int32)


@triton.jit
def processHistogramStep(
    row_ptr,
    stride_xn,
    row_start,
    row_end,
    seq_len,
    step_idx: tl.constexpr,
    logit_pattern,
    s_step_thresholds_ptr,
    found_topk_values,
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024
    RADIX10_MASK: tl.constexpr = 0x3FF

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    for clear_round in tl.range(0, RADIX11_SIZE // BLOCK_SIZE):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + clear_bins, 0)
    tl.debug_barrier()

    if step_idx == 2:
        step1_threshold = tl.load(s_step_thresholds_ptr + 1)
        logit_pattern = (step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21
    elif step_idx == 3:
        step1_threshold = tl.load(s_step_thresholds_ptr + 1)
        step2_threshold = tl.load(s_step_thresholds_ptr + 2)
        logit_pattern = ((step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21) | (
            (step2_threshold.to(tl.uint32) & RADIX11_MASK) << 10)

    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    n_vec_full = seq_len // (BLOCK_SIZE * VEC)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

    if ASSUME_ALIGNED:
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x_vec)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            tl.atomic_add(
                hist_base_ptr + digit,
                ones_vec_2d,
                mask=partial,
                sem="relaxed",
                scope="cta",
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            key = _convert_to_trt_uint32(x)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            tl.atomic_add(
                hist_base_ptr + digit,
                ones,
                mask=partial,
                sem="relaxed",
                scope="cta",
            )
    else:
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
            key = _convert_to_trt_uint32(x)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = in_range
            elif step_idx == 2:
                partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
            else:
                partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

            tl.atomic_add(
                hist_base_ptr + digit,
                ones,
                mask=partial,
                sem="relaxed",
                scope="cta",
            )
    tl.debug_barrier()

    # TRT-style threshold search with per-round early-exit.
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.debug_barrier()
    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    last_value = found_topk_values
    threshold_found = False
    threshold_rounds = tl.where(
        step_idx == 3,
        RADIX10_SIZE // BLOCK_SIZE,
        RADIX11_SIZE // BLOCK_SIZE,
    )
    for round_idx in tl.range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(hist_base_ptr + bins)
            prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    tl.store(s_step_thresholds_ptr + step_idx, threshold_bin_idx)

    use_final = (step_idx < 3) & (threshold_bin_idx >= 0) & (final_bin_size <= FINAL_SORT_ITEMS)
    if use_final:
        tl.store(s_final_cnt_ptr, 0)

    found_ptrs = s_found_topk_values_ptr + zeros
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    if ASSUME_ALIGNED:
        found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            offs = base[:, None] + vec[None, :]
            x_vec = tl.load(row_ptr + offs)
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x_vec)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            take_lt = partial & (digit < threshold_bin_idx)
            out_pos_lt = tl.atomic_add(
                found_ptrs_vec_2d,
                ones_vec_2d,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_lt,
                offs.to(tl.int32),
                mask=take_lt & (out_pos_lt < TOPK),
            )

            if step_idx == 3:
                take_eq = partial & (digit == threshold_bin_idx)
                out_pos_eq = tl.atomic_add(
                    hist_base_ptr + digit,
                    ones_vec_2d,
                    mask=take_eq,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_eq,
                    offs.to(tl.int32),
                    mask=take_eq & (out_pos_eq < TOPK),
                )
            elif use_final:
                take_eq_final = partial & (digit == threshold_bin_idx)
                final_pos = tl.atomic_add(
                    final_cnt_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_eq_final,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    hist_base_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
                tl.store(
                    hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                    x_vec.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )

        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
            x = tl.load(row_ptr + offs)
            key = _convert_to_trt_uint32(x)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = tl.full([BLOCK_SIZE], True, tl.int1)
            elif step_idx == 2:
                partial = ((key ^ logit_pattern) >> 21) == 0
            else:
                partial = ((key ^ logit_pattern) >> 10) == 0

            take_lt = partial & (digit < threshold_bin_idx)
            out_pos_lt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_lt,
                offs.to(tl.int32),
                mask=take_lt & (out_pos_lt < TOPK),
            )

            if step_idx == 3:
                take_eq = partial & (digit == threshold_bin_idx)
                out_pos_eq = tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=take_eq,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_eq,
                    offs.to(tl.int32),
                    mask=take_eq & (out_pos_eq < TOPK),
                )
            elif use_final:
                take_eq_final = partial & (digit == threshold_bin_idx)
                final_pos = tl.atomic_add(
                    final_cnt_ptrs,
                    ones,
                    mask=take_eq_final,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    hist_base_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
                tl.store(
                    hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                    x.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
    else:
        for t in tl.range(0, n_tiles):
            offs = t * BLOCK_SIZE + lane
            in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
            x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
            key = _convert_to_trt_uint32(x)
            if step_idx == 0:
                digit = _convert_to_trt_uint16_hi11(x)
            elif step_idx == 1:
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
            elif step_idx == 2:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            if step_idx < 2:
                partial = in_range
            elif step_idx == 2:
                partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
            else:
                partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

            take_lt = partial & (digit < threshold_bin_idx)
            out_pos_lt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_lt,
                offs.to(tl.int32),
                mask=take_lt & (out_pos_lt < TOPK),
            )

            if step_idx == 3:
                take_eq = partial & (digit == threshold_bin_idx)
                out_pos_eq = tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=take_eq,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_eq,
                    offs.to(tl.int32),
                    mask=take_eq & (out_pos_eq < TOPK),
                )
            elif use_final:
                take_eq_final = partial & (digit == threshold_bin_idx)
                final_pos = tl.atomic_add(
                    final_cnt_ptrs,
                    ones,
                    mask=take_eq_final,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    hist_base_ptr + final_pos,
                    offs.to(tl.int32),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
                tl.store(
                    hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                    x.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )

    if step_idx < 3:
        if use_final:
            need_final_sort = True
            continue_to_next_step = False
        else:
            need_final_sort = False
            continue_to_next_step = True
    else:
        tl.store(s_found_topk_values_ptr, TOPK)
        need_final_sort = False
        continue_to_next_step = False

    tl.debug_barrier()
    return continue_to_next_step, need_final_sort, logit_pattern


@triton.jit
def _tle_final_select_radix(
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FINAL_SORT_ITEMS: tl.constexpr,
):
    RADIX_BITS_FINAL: tl.constexpr = 8
    RADIX_SIZE_FINAL: tl.constexpr = 1 << RADIX_BITS_FINAL
    RADIX_MASK_FINAL: tl.constexpr = RADIX_SIZE_FINAL - 1
    DIGIT_START: tl.constexpr = 32 - RADIX_BITS_FINAL

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    bins = tl.arange(0, RADIX_SIZE_FINAL)

    s_radix_counts = tle.gpu.alloc(
        [RADIX_SIZE_FINAL],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    radix_count_ptr = tle.gpu.local_ptr(s_radix_counts, (0, ))
    radix_count_vec_ptr = tle.gpu.local_ptr(s_radix_counts, (bins, ))

    base_idx = tl.load(s_found_topk_values_ptr)
    final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
    remain = tl.minimum(TOPK - base_idx, final_cnt)
    if remain > 0:
        desired = tl.zeros((), dtype=tl.uint32)
        desired_mask = tl.zeros((), dtype=tl.uint32)
        k_to_find = remain + 1

        for digit_pos in tl.static_range(DIGIT_START, -1, -RADIX_BITS_FINAL):
            tl.store(radix_count_ptr + lane, 0, mask=lane < RADIX_SIZE_FINAL)
            tl.debug_barrier()

            cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
            for t in tl.range(0, cnt_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < final_cnt
                x_bits_i32 = tl.load(
                    hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                    mask=valid,
                    other=0,
                )
                x = x_bits_i32.to(tl.float32, bitcast=True)
                key = _convert_to_trt_uint32(x)
                matches = (key & desired_mask) == desired
                digit = ((key >> digit_pos) & RADIX_MASK_FINAL).to(tl.int32)
                take = valid & matches
                tl.atomic_add(
                    radix_count_ptr + digit,
                    ones,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )

            tl.debug_barrier()
            counts = tl.load(radix_count_vec_ptr)
            prefix_sum, _ = tle.cumsum(counts, axis=0, reverse=False)
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < k_to_find) & (next_prefix_sum >= k_to_find)
            threshold_init = tl.full((), RADIX_SIZE_FINAL, dtype=tl.int32)
            threshold_bin = tl.min(tl.where(threshold_mask, bins, threshold_init), axis=0).to(tl.int32)
            threshold_bin = tl.where(threshold_bin == RADIX_SIZE_FINAL, RADIX_SIZE_FINAL - 1, threshold_bin)
            counts_lt = tl.max(tl.where(bins == threshold_bin, prefix_sum, 0), axis=0).to(tl.int32)

            desired = desired | (threshold_bin.to(tl.uint32) << digit_pos)
            desired_mask = desired_mask | (tl.full((), RADIX_MASK_FINAL, dtype=tl.uint32) << digit_pos)
            k_to_find = k_to_find - counts_lt

        thr_key = desired
        found_ptrs = s_found_topk_values_ptr + zeros
        cnt_tiles = tl.cdiv(final_cnt, BLOCK_SIZE)
        for t in tl.range(0, cnt_tiles):
            pos = t * BLOCK_SIZE + lane
            valid = pos < final_cnt
            idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
            x_bits_i32 = tl.load(
                hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                mask=valid,
                other=0,
            )
            x = x_bits_i32.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x)
            take_lt = valid & (key < thr_key)
            out_pos_gt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_gt,
                idx,
                mask=take_lt & (out_pos_gt < TOPK),
            )

        cur = tl.load(s_found_topk_values_ptr)
        if cur < TOPK:
            for t in tl.range(0, cnt_tiles):
                cur = tl.load(s_found_topk_values_ptr)
                if cur < TOPK:
                    pos = t * BLOCK_SIZE + lane
                    valid = pos < final_cnt
                    idx = tl.load(hist_base_ptr + pos, mask=valid, other=0)
                    x_bits_i32 = tl.load(
                        hist_base_ptr + (FINAL_SORT_ITEMS + pos),
                        mask=valid,
                        other=0,
                    )
                    x = x_bits_i32.to(tl.float32, bitcast=True)
                    key = _convert_to_trt_uint32(x)
                    take_eq = valid & (key == thr_key)
                    out_pos_eq = tl.atomic_add(
                        found_ptrs,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_ptr + out_pos_eq,
                        idx,
                        mask=take_eq & (out_pos_eq < TOPK),
                    )

    tl.store(s_found_topk_values_ptr, TOPK)


# %%
# TLE kernel (shared memory)
# --------------------------


@triton.jit
def tle_topk_selector_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    row_len = row_end - row_start

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            tl.store(out_row + pos * stride_outn, (row_start + pos).to(tl.int32), mask=take_row)
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_row + pos * stride_outn, -1, mask=take_pad)
        return

    FINAL_SORT_ITEMS: tl.constexpr = 2048
    HIST_SIZE: tl.constexpr = 4096

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hist_base_ptr = tle.gpu.local_ptr(s_histogram, (0, ))
    # TRT-style union reuse:
    # - [0, FINAL_SORT_ITEMS): final indices (int32)
    # - [FINAL_SORT_ITEMS, 2*FINAL_SORT_ITEMS): final logits bitcast(int32)
    s_out_indices = tle.gpu.alloc(
        [TOPK],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_step_thresholds = tle.gpu.alloc(
        [4],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0, ))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0, ))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0, ))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0, ))
    s_step_thresholds_ptr = tle.gpu.local_ptr(s_step_thresholds, (0, ))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0, ))
    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)

    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = True
    need_final_sort = False
    init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for init_idx in tl.range(0, init_chunks):
        pos = init_idx * BLOCK_SIZE + lane
        tl.store(tle.gpu.local_ptr(s_out_indices, (pos, )), -1, mask=pos < TOPK)

    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            found_topk_values = tl.load(s_found_topk_values_ptr)
            continue_to_next_step, step_need_final_sort, logit_pattern = processHistogramStep(
                row_ptr,
                stride_xn,
                row_start,
                row_end,
                seq_len,
                step_idx,
                logit_pattern,
                s_step_thresholds_ptr,
                found_topk_values,
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_threshold_bin_idx_ptr,
                s_final_bin_size_ptr,
                ASSUME_ALIGNED=ASSUME_ALIGNED,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            need_final_sort = need_final_sort | step_need_final_sort

    if need_final_sort:
        if USE_RADIX_FINAL:
            _tle_final_select_radix(
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            # Guard against stale/oversized counts to avoid out-of-bounds accesses
            # in the shared-memory final buffers.
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + pos, )),
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + j, )))
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(
                    tle.gpu.local_ptr(s_histogram, (pos, )),
                    mask=take,
                    other=0,
                )
                tl.store(tle.gpu.local_ptr(s_out_indices, (dst_pos, )), idx_i, mask=take)
            tl.store(s_found_topk_values_ptr, TOPK)

    flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < TOPK
        out_vals = tl.load(tle.gpu.local_ptr(s_out_indices, (pos, )), mask=mask, other=-1)
        tl.store(out_row + pos * stride_outn, out_vals, mask=mask)


@triton.jit
def _tle_topk_smem_overflow_fallback_fullscan(
    row_ptr,
    out_row,
    stride_xn,
    stride_outn,
    row_start,
    row_end,
    seq_len,
    hist_base_ptr,
    s_write_count_ptr,
    s_eq_count_ptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX_SIZE: tl.constexpr = 256
    CAND_ROUNDS: tl.constexpr = 4

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    out_init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    hist_clear_chunks: tl.constexpr = (RADIX_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_scan_tiles = tl.cdiv(seq_len, BLOCK_SIZE)

    tl.store(s_write_count_ptr, 0)
    tl.store(s_eq_count_ptr, 0)
    for t in tl.range(0, out_init_chunks):
        pos = t * BLOCK_SIZE + lane
        tl.store(out_row + pos * stride_outn, -1, mask=pos < TOPK)

    for t in tl.range(0, hist_clear_chunks):
        bins = t * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + bins, 0, mask=bins < RADIX_SIZE)
    tl.debug_barrier()

    for t in tl.range(0, num_scan_tiles):
        offs = t * BLOCK_SIZE + lane
        valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
        digit = _convert_to_uint16_hi8(x)
        tl.atomic_add(
            hist_base_ptr + digit,
            ones,
            mask=valid,
            sem="relaxed",
            scope="cta",
        )
    tl.debug_barrier()

    radix_bins = tl.arange(0, RADIX_SIZE)
    zeros_radix = tl.zeros([RADIX_SIZE], dtype=tl.int32)
    counts = tl.load(hist_base_ptr + radix_bins)
    gt_exclusive, _ = tle.cumsum(counts, axis=0, reverse=True)
    cumsum_desc = gt_exclusive + counts
    threshold_mask = (cumsum_desc >= TOPK) & (gt_exclusive < TOPK)
    coarse_threshold_bin = tl.sum(
        tl.where(threshold_mask, radix_bins, zeros_radix),
        axis=0,
    )
    coarse_counts_gt = tl.sum(
        tl.where(threshold_mask, gt_exclusive, zeros_radix),
        axis=0,
    )
    gt_cursors = tl.where(radix_bins > coarse_threshold_bin, gt_exclusive, zeros_radix)
    tl.store(hist_base_ptr + radix_bins, gt_cursors)
    remaining = TOPK - coarse_counts_gt
    tl.store(s_write_count_ptr + zeros, coarse_counts_gt)
    tl.debug_barrier()

    for t in tl.range(0, num_scan_tiles):
        offs = t * BLOCK_SIZE + lane
        valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        idx = offs.to(tl.int32)
        x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
        digit = _convert_to_uint16_hi8(x)

        take_gt = valid & (digit > coarse_threshold_bin)
        out_pos_gt = tl.atomic_add(
            hist_base_ptr + digit,
            ones,
            mask=take_gt,
            sem="relaxed",
            scope="cta",
        )
        tl.store(
            out_row + out_pos_gt * stride_outn,
            idx,
            mask=take_gt & (out_pos_gt < TOPK),
        )
    tl.debug_barrier()

    refine_prefix = tl.zeros((), dtype=tl.uint32)
    refine_mask = tl.zeros((), dtype=tl.uint32)
    for round_idx in tl.static_range(CAND_ROUNDS):
        if remaining > 0:
            for t in tl.range(0, hist_clear_chunks):
                bins = t * BLOCK_SIZE + lane
                tl.store(hist_base_ptr + bins, 0, mask=bins < RADIX_SIZE)
            tl.debug_barrier()

            shift: tl.constexpr = 24 - round_idx * 8
            for t in tl.range(0, num_scan_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                coarse_digit = _convert_to_uint16_hi8(x)
                ordered = _convert_to_uint32(x)
                prefix_match = (ordered & refine_mask) == refine_prefix
                active = valid & (coarse_digit == coarse_threshold_bin) & prefix_match
                digit = ((ordered >> shift) & 0xFF).to(tl.int32)
                tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=active,
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()

            radix_bins = tl.arange(0, RADIX_SIZE)
            zeros_radix = tl.zeros([RADIX_SIZE], dtype=tl.int32)
            counts = tl.load(hist_base_ptr + radix_bins)
            gt_exclusive, _ = tle.cumsum(counts, axis=0, reverse=True)
            cumsum_desc = gt_exclusive + counts
            base_write = tl.load(s_write_count_ptr)
            threshold_mask = (cumsum_desc >= remaining) & (gt_exclusive < remaining)
            threshold_bin = tl.sum(
                tl.where(threshold_mask, radix_bins, zeros_radix),
                axis=0,
            )
            counts_gt = tl.sum(
                tl.where(threshold_mask, gt_exclusive, zeros_radix),
                axis=0,
            )
            gt_cursors = tl.where(
                radix_bins > threshold_bin,
                base_write + gt_exclusive,
                zeros_radix,
            )
            tl.store(hist_base_ptr + radix_bins, gt_cursors)
            remaining = remaining - counts_gt
            tl.store(s_write_count_ptr + zeros, base_write + counts_gt)
            if round_idx == (CAND_ROUNDS - 1):
                tl.store(s_eq_count_ptr, 0)
            tl.debug_barrier()

            for t in tl.range(0, num_scan_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                idx = offs.to(tl.int32)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                coarse_digit = _convert_to_uint16_hi8(x)
                ordered = _convert_to_uint32(x)
                prefix_match = (ordered & refine_mask) == refine_prefix
                active = valid & (coarse_digit == coarse_threshold_bin) & prefix_match
                digit = ((ordered >> shift) & 0xFF).to(tl.int32)

                take_gt = active & (digit > threshold_bin)
                out_pos_gt = tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=take_gt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    out_row + out_pos_gt * stride_outn,
                    idx,
                    mask=take_gt & (out_pos_gt < TOPK),
                )

                if remaining > 0:
                    take_eq = active & (digit == threshold_bin)
                    if round_idx == (CAND_ROUNDS - 1):
                        eq_pos = tl.atomic_add(
                            s_eq_count_ptr + zeros,
                            ones,
                            mask=take_eq,
                            sem="relaxed",
                            scope="cta",
                        )
                        take_eq_select = take_eq & (eq_pos < remaining)
                        out_pos_eq = tl.atomic_add(
                            s_write_count_ptr + zeros,
                            ones,
                            mask=take_eq_select,
                            sem="relaxed",
                            scope="cta",
                        )
                        tl.store(
                            out_row + out_pos_eq * stride_outn,
                            idx,
                            mask=take_eq_select & (out_pos_eq < TOPK),
                        )
            tl.debug_barrier()

            threshold_u32 = threshold_bin.to(tl.uint32)
            if round_idx == 0:
                refine_prefix = threshold_u32 << 24
                refine_mask = tl.full((), 0xFF000000, tl.uint32)
            elif round_idx == 1:
                refine_prefix = refine_prefix | (threshold_u32 << 16)
                refine_mask = tl.full((), 0xFFFF0000, tl.uint32)
            elif round_idx == 2:
                refine_prefix = refine_prefix | (threshold_u32 << 8)
                refine_mask = tl.full((), 0xFFFFFF00, tl.uint32)


@triton.jit
def _tle_process_histogram_step_smem(
    row_ptr,
    stride_xn,
    row_start,
    row_end,
    seq_len,
    step_idx: tl.constexpr,
    found_topk_values,
    hist_base_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    src_idx_ptr,
    src_val_ptr,
    src_count_ptr,
    dst_idx_ptr,
    dst_val_ptr,
    dst_count_ptr,
    s_need_fallback_ptr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SMEM_INPUT_SIZE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024
    RADIX10_MASK: tl.constexpr = 0x3FF

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)

    clear_rounds = tl.where(step_idx == 2, RADIX10_SIZE // BLOCK_SIZE, RADIX11_SIZE // BLOCK_SIZE)
    threshold_rounds = clear_rounds

    for clear_round in tl.range(0, clear_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(hist_base_ptr + clear_bins, 0)

    if step_idx == 0:
        if ASSUME_ALIGNED:
            n_vec_full = seq_len // (BLOCK_SIZE * VEC)
            rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

            for t in tl.range(0, n_vec_full):
                base = t * BLOCK_SIZE * VEC + lane * VEC
                offs = base[:, None] + vec[None, :]
                x_vec = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x_vec)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                tl.atomic_add(
                    hist_base_ptr + digit,
                    ones_vec_2d,
                    sem="relaxed",
                    scope="cta",
                )

            for t in tl.range(0, rem_tiles):
                offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
                x = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    sem="relaxed",
                    scope="cta",
                )
        else:
            n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
            for t in tl.range(0, n_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                key = _convert_to_trt_uint32(x)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                tl.atomic_add(
                    hist_base_ptr + digit,
                    ones,
                    mask=valid,
                    sem="relaxed",
                    scope="cta",
                )
    else:
        src_count = tl.minimum(tl.load(src_count_ptr), SMEM_INPUT_SIZE)
        src_vec_full = src_count // (BLOCK_SIZE * VEC)
        vec_processed = src_vec_full * BLOCK_SIZE * VEC
        src_tail_tiles = tl.cdiv(src_count - vec_processed, BLOCK_SIZE)

        for t in tl.range(0, src_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            pos = base[:, None] + vec[None, :]
            val_bits_vec = tl.load(src_val_ptr + pos)
            x_vec = val_bits_vec.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 1:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)
            tl.atomic_add(
                hist_base_ptr + digit,
                ones_vec_2d,
                sem="relaxed",
                scope="cta",
            )

        for t in tl.range(0, src_tail_tiles):
            pos = vec_processed + t * BLOCK_SIZE + lane
            valid = pos < src_count
            val_bits = tl.load(src_val_ptr + pos, mask=valid, other=0)
            x = val_bits.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x)
            if step_idx == 1:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)
            tl.atomic_add(
                hist_base_ptr + digit,
                ones,
                mask=valid,
                sem="relaxed",
                scope="cta",
            )
    tl.debug_barrier()

    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    last_value = found_topk_values
    threshold_found = False
    for round_idx in tl.range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(hist_base_ptr + bins)
            prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
    use_final = (step_idx < 2) & (threshold_bin_idx >= 0) & (final_bin_size <= FINAL_SORT_ITEMS)
    if use_final:
        tl.store(s_final_cnt_ptr, 0)
    elif step_idx < 2:
        tl.store(dst_count_ptr, 0)

    found_ptrs = s_found_topk_values_ptr + zeros
    found_ptrs_vec_2d = s_found_topk_values_ptr + zeros_vec_2d
    final_cnt_ptrs = s_final_cnt_ptr + zeros
    final_cnt_ptrs_vec_2d = s_final_cnt_ptr + zeros_vec_2d
    dst_count_ptrs = dst_count_ptr + zeros
    dst_count_ptrs_vec_2d = dst_count_ptr + zeros_vec_2d
    fallback_ptrs = s_need_fallback_ptr + zeros
    fallback_ptrs_vec_2d = s_need_fallback_ptr + zeros_vec_2d

    if step_idx == 0:
        if ASSUME_ALIGNED:
            n_vec_full = seq_len // (BLOCK_SIZE * VEC)
            rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

            for t in tl.range(0, n_vec_full):
                base = t * BLOCK_SIZE * VEC + lane * VEC
                offs = base[:, None] + vec[None, :]
                idx = offs.to(tl.int32)
                x_vec = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x_vec)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)

                take_lt = digit < threshold_bin_idx
                out_pos_lt = tl.atomic_add(
                    found_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_lt,
                    idx,
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if use_final:
                    take_eq_final = digit == threshold_bin_idx
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs_vec_2d,
                        ones_vec_2d,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        hist_base_ptr + final_pos,
                        idx,
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x_vec.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                elif step_idx < 2:
                    take_eq_next = digit == threshold_bin_idx
                    dst_pos = tl.atomic_add(
                        dst_count_ptrs_vec_2d,
                        ones_vec_2d,
                        mask=take_eq_next,
                        sem="relaxed",
                        scope="cta",
                    )
                    keep_eq = take_eq_next & (dst_pos < SMEM_INPUT_SIZE)
                    tl.store(dst_idx_ptr + dst_pos, idx, mask=keep_eq)
                    tl.store(dst_val_ptr + dst_pos, x_vec.to(tl.int32, bitcast=True), mask=keep_eq)
                    overflow_mask = take_eq_next & (dst_pos >= SMEM_INPUT_SIZE)
                    tl.atomic_or(
                        fallback_ptrs_vec_2d,
                        ones_vec_2d,
                        mask=overflow_mask,
                        sem="relaxed",
                        scope="cta",
                    )

            for t in tl.range(0, rem_tiles):
                offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
                idx = offs.to(tl.int32)
                x = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)

                take_lt = digit < threshold_bin_idx
                out_pos_lt = tl.atomic_add(
                    found_ptrs,
                    ones,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_lt,
                    idx,
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if use_final:
                    take_eq_final = digit == threshold_bin_idx
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs,
                        ones,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        hist_base_ptr + final_pos,
                        idx,
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                elif step_idx < 2:
                    take_eq_next = digit == threshold_bin_idx
                    dst_pos = tl.atomic_add(
                        dst_count_ptrs,
                        ones,
                        mask=take_eq_next,
                        sem="relaxed",
                        scope="cta",
                    )
                    keep_eq = take_eq_next & (dst_pos < SMEM_INPUT_SIZE)
                    tl.store(dst_idx_ptr + dst_pos, idx, mask=keep_eq)
                    tl.store(dst_val_ptr + dst_pos, x.to(tl.int32, bitcast=True), mask=keep_eq)
                    overflow_mask = take_eq_next & (dst_pos >= SMEM_INPUT_SIZE)
                    tl.atomic_or(
                        fallback_ptrs,
                        ones,
                        mask=overflow_mask,
                        sem="relaxed",
                        scope="cta",
                    )
        else:
            n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
            for t in tl.range(0, n_tiles):
                offs = t * BLOCK_SIZE + lane
                valid = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                idx = offs.to(tl.int32)
                x = tl.load(row_ptr + offs * stride_xn, mask=valid, other=float("-inf"))
                key = _convert_to_trt_uint32(x)
                digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)

                take_lt = valid & (digit < threshold_bin_idx)
                out_pos_lt = tl.atomic_add(
                    found_ptrs,
                    ones,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_lt,
                    idx,
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if use_final:
                    take_eq_final = valid & (digit == threshold_bin_idx)
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs,
                        ones,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        hist_base_ptr + final_pos,
                        idx,
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                elif step_idx < 2:
                    take_eq_next = valid & (digit == threshold_bin_idx)
                    dst_pos = tl.atomic_add(
                        dst_count_ptrs,
                        ones,
                        mask=take_eq_next,
                        sem="relaxed",
                        scope="cta",
                    )
                    keep_eq = take_eq_next & (dst_pos < SMEM_INPUT_SIZE)
                    tl.store(dst_idx_ptr + dst_pos, idx, mask=keep_eq)
                    tl.store(dst_val_ptr + dst_pos, x.to(tl.int32, bitcast=True), mask=keep_eq)
                    overflow_mask = take_eq_next & (dst_pos >= SMEM_INPUT_SIZE)
                    tl.atomic_or(
                        fallback_ptrs,
                        ones,
                        mask=overflow_mask,
                        sem="relaxed",
                        scope="cta",
                    )
    else:
        src_count = tl.minimum(tl.load(src_count_ptr), SMEM_INPUT_SIZE)
        src_vec_full = src_count // (BLOCK_SIZE * VEC)
        vec_processed = src_vec_full * BLOCK_SIZE * VEC
        src_tail_tiles = tl.cdiv(src_count - vec_processed, BLOCK_SIZE)

        for t in tl.range(0, src_vec_full):
            base = t * BLOCK_SIZE * VEC + lane * VEC
            pos = base[:, None] + vec[None, :]
            idx = tl.load(src_idx_ptr + pos)
            val_bits_vec = tl.load(src_val_ptr + pos)
            x_vec = val_bits_vec.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x_vec)
            if step_idx == 1:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            take_lt = digit < threshold_bin_idx
            out_pos_lt = tl.atomic_add(
                found_ptrs_vec_2d,
                ones_vec_2d,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_lt,
                idx.to(tl.int32),
                mask=take_lt & (out_pos_lt < TOPK),
            )

            if step_idx == 2:
                take_eq = digit == threshold_bin_idx
                out_pos_eq = tl.atomic_add(
                    found_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_eq,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_eq,
                    idx.to(tl.int32),
                    mask=take_eq & (out_pos_eq < TOPK),
                )
            elif use_final:
                take_eq_final = digit == threshold_bin_idx
                final_pos = tl.atomic_add(
                    final_cnt_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_eq_final,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    hist_base_ptr + final_pos,
                    idx.to(tl.int32),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
                tl.store(
                    hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                    x_vec.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
            elif step_idx < 2:
                take_eq_next = digit == threshold_bin_idx
                dst_pos = tl.atomic_add(
                    dst_count_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_eq_next,
                    sem="relaxed",
                    scope="cta",
                )
                keep_eq = take_eq_next & (dst_pos < SMEM_INPUT_SIZE)
                tl.store(dst_idx_ptr + dst_pos, idx.to(tl.int32), mask=keep_eq)
                tl.store(dst_val_ptr + dst_pos, val_bits_vec, mask=keep_eq)
                overflow_mask = take_eq_next & (dst_pos >= SMEM_INPUT_SIZE)
                tl.atomic_or(
                    fallback_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=overflow_mask,
                    sem="relaxed",
                    scope="cta",
                )

        for t in tl.range(0, src_tail_tiles):
            pos = vec_processed + t * BLOCK_SIZE + lane
            valid = pos < src_count
            idx = tl.load(src_idx_ptr + pos, mask=valid, other=0)
            val_bits = tl.load(src_val_ptr + pos, mask=valid, other=0)
            x = val_bits.to(tl.float32, bitcast=True)
            key = _convert_to_trt_uint32(x)
            if step_idx == 1:
                digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
            else:
                digit = (key & RADIX10_MASK).to(tl.int32)

            take_lt = valid & (digit < threshold_bin_idx)
            out_pos_lt = tl.atomic_add(
                found_ptrs,
                ones,
                mask=take_lt,
                sem="relaxed",
                scope="cta",
            )
            tl.store(
                s_out_indices_ptr + out_pos_lt,
                idx.to(tl.int32),
                mask=take_lt & (out_pos_lt < TOPK),
            )

            if step_idx == 2:
                take_eq = valid & (digit == threshold_bin_idx)
                out_pos_eq = tl.atomic_add(
                    found_ptrs,
                    ones,
                    mask=take_eq,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_ptr + out_pos_eq,
                    idx.to(tl.int32),
                    mask=take_eq & (out_pos_eq < TOPK),
                )
            elif use_final:
                take_eq_final = valid & (digit == threshold_bin_idx)
                final_pos = tl.atomic_add(
                    final_cnt_ptrs,
                    ones,
                    mask=take_eq_final,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    hist_base_ptr + final_pos,
                    idx.to(tl.int32),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
                tl.store(
                    hist_base_ptr + (FINAL_SORT_ITEMS + final_pos),
                    x.to(tl.int32, bitcast=True),
                    mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                )
            elif step_idx < 2:
                take_eq_next = valid & (digit == threshold_bin_idx)
                dst_pos = tl.atomic_add(
                    dst_count_ptrs,
                    ones,
                    mask=take_eq_next,
                    sem="relaxed",
                    scope="cta",
                )
                keep_eq = take_eq_next & (dst_pos < SMEM_INPUT_SIZE)
                tl.store(dst_idx_ptr + dst_pos, idx.to(tl.int32), mask=keep_eq)
                tl.store(dst_val_ptr + dst_pos, val_bits, mask=keep_eq)
                overflow_mask = take_eq_next & (dst_pos >= SMEM_INPUT_SIZE)
                tl.atomic_or(
                    fallback_ptrs,
                    ones,
                    mask=overflow_mask,
                    sem="relaxed",
                    scope="cta",
                )

    tl.debug_barrier()
    if step_idx < 2:
        if use_final:
            continue_to_next_step = False
            need_final_sort = True
        else:
            continue_to_next_step = True
            need_final_sort = False
    else:
        tl.store(s_found_topk_values_ptr, TOPK)
        continue_to_next_step = False
        need_final_sort = False
    return continue_to_next_step, need_final_sort


@triton.jit
def tle_topk_selector_kernel_smem(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SMEM_INPUT_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    row_len = row_end - row_start

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    if row_len <= TOPK:
        chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
        for chunk_idx in tl.range(0, chunks):
            pos = chunk_idx * BLOCK_SIZE + lane
            take_row = pos < row_len
            tl.store(out_row + pos * stride_outn, (row_start + pos).to(tl.int32), mask=take_row)
            take_pad = (pos >= row_len) & (pos < TOPK)
            tl.store(out_row + pos * stride_outn, -1, mask=take_pad)
        return

    FINAL_SORT_ITEMS: tl.constexpr = 2048
    HIST_SIZE: tl.constexpr = 4096

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hist_base_ptr = tle.gpu.local_ptr(s_histogram, (0, ))
    s_out_indices = tle.gpu.alloc(
        [TOPK],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_idx0 = tle.gpu.alloc(
        [SMEM_INPUT_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_idx1 = tle.gpu.alloc(
        [SMEM_INPUT_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_val0 = tle.gpu.alloc(
        [SMEM_INPUT_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_val1 = tle.gpu.alloc(
        [SMEM_INPUT_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_count0 = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_input_count1 = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_need_fallback = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_fallback_eq_count = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    s_input_idx0_ptr = tle.gpu.local_ptr(s_input_idx0, (0, ))
    s_input_idx1_ptr = tle.gpu.local_ptr(s_input_idx1, (0, ))
    s_input_val0_ptr = tle.gpu.local_ptr(s_input_val0, (0, ))
    s_input_val1_ptr = tle.gpu.local_ptr(s_input_val1, (0, ))
    s_input_count0_ptr = tle.gpu.local_ptr(s_input_count0, (0, ))
    s_input_count1_ptr = tle.gpu.local_ptr(s_input_count1, (0, ))
    s_need_fallback_ptr = tle.gpu.local_ptr(s_need_fallback, (0, ))
    s_fallback_eq_count_ptr = tle.gpu.local_ptr(s_fallback_eq_count, (0, ))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0, ))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0, ))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0, ))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0, ))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0, ))

    tl.store(s_input_count0_ptr, 0)
    tl.store(s_input_count1_ptr, 0)
    tl.store(s_need_fallback_ptr, 0)
    tl.store(s_fallback_eq_count_ptr, 0)
    tl.store(s_final_cnt_ptr, 0)
    tl.store(s_threshold_bin_idx_ptr, -1)
    tl.store(s_final_bin_size_ptr, 0)
    tl.store(s_found_topk_values_ptr, 0)

    init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for init_idx in tl.range(0, init_chunks):
        pos = init_idx * BLOCK_SIZE + lane
        tl.store(s_out_indices_ptr + pos, -1, mask=pos < TOPK)

    continue_to_next_step = True
    need_final_sort = False
    for step_idx in tl.static_range(0, 3):
        if continue_to_next_step:
            found_topk_values = tl.load(s_found_topk_values_ptr)
            if step_idx == 0:
                src_idx_ptr = s_input_idx0_ptr
                src_val_ptr = s_input_val0_ptr
                src_count_ptr = s_input_count0_ptr
                dst_idx_ptr = s_input_idx0_ptr
                dst_val_ptr = s_input_val0_ptr
                dst_count_ptr = s_input_count0_ptr
            elif step_idx == 1:
                src_idx_ptr = s_input_idx0_ptr
                src_val_ptr = s_input_val0_ptr
                src_count_ptr = s_input_count0_ptr
                dst_idx_ptr = s_input_idx1_ptr
                dst_val_ptr = s_input_val1_ptr
                dst_count_ptr = s_input_count1_ptr
            else:
                src_idx_ptr = s_input_idx1_ptr
                src_val_ptr = s_input_val1_ptr
                src_count_ptr = s_input_count1_ptr
                dst_idx_ptr = s_input_idx0_ptr
                dst_val_ptr = s_input_val0_ptr
                dst_count_ptr = s_input_count0_ptr

            continue_to_next_step, step_need_final_sort = _tle_process_histogram_step_smem(
                row_ptr,
                stride_xn,
                row_start,
                row_end,
                seq_len,
                step_idx,
                found_topk_values,
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_threshold_bin_idx_ptr,
                s_final_bin_size_ptr,
                src_idx_ptr,
                src_val_ptr,
                src_count_ptr,
                dst_idx_ptr,
                dst_val_ptr,
                dst_count_ptr,
                s_need_fallback_ptr,
                ASSUME_ALIGNED=ASSUME_ALIGNED,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                SMEM_INPUT_SIZE=SMEM_INPUT_SIZE,
            )
            need_final_sort = need_final_sort | step_need_final_sort

    if tl.load(s_need_fallback_ptr) != 0:
        _tle_topk_smem_overflow_fallback_fullscan(
            row_ptr,
            out_row,
            stride_xn,
            stride_outn,
            row_start,
            row_end,
            seq_len,
            hist_base_ptr,
            s_found_topk_values_ptr,
            s_fallback_eq_count_ptr,
            TOPK=TOPK,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return

    if need_final_sort:
        if USE_RADIX_FINAL:
            _tle_final_select_radix(
                hist_base_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + pos, )),
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + j, )))
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(
                    tle.gpu.local_ptr(s_histogram, (pos, )),
                    mask=take,
                    other=0,
                )
                tl.store(tle.gpu.local_ptr(s_out_indices, (dst_pos, )), idx_i, mask=take)
            tl.store(s_found_topk_values_ptr, TOPK)

    flush_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for flush_chunk in tl.static_range(flush_chunks):
        pos = flush_chunk * BLOCK_SIZE + lane
        mask = pos < TOPK
        out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
        tl.store(out_row + pos * stride_outn, out_vals, mask=mask)


@triton.jit
def _tle_process_histogram_step_cluster(
    row_ptr,
    stride_xn,
    row_start,
    row_end,
    seq_len,
    step_idx: tl.constexpr,
    logit_pattern,
    cluster_rank,
    is_rank0,
    s_step_local_hist_ptr,
    s_histogram_ptr,
    s_histogram_rank0_ptr,
    s_out_indices_ptr,
    s_final_cnt_ptr,
    s_found_topk_values_ptr,
    s_step_thresholds_ptr,
    s_step_thresholds_rank0_ptr,
    s_threshold_bin_idx_ptr,
    s_final_bin_size_ptr,
    s_threshold_bin_idx_rank0_ptr,
    s_final_bin_size_rank0_ptr,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    VEC: tl.constexpr = 4
    FINAL_SORT_ITEMS: tl.constexpr = 2048
    RADIX11_SIZE: tl.constexpr = 2048
    RADIX11_MASK: tl.constexpr = 0x7FF
    RADIX10_SIZE: tl.constexpr = 1024
    RADIX10_MASK: tl.constexpr = 0x3FF

    lane = tl.arange(0, BLOCK_SIZE)
    vec = tl.arange(0, VEC)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)
    ones_vec_2d = tl.full([BLOCK_SIZE, VEC], 1, tl.int32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    zeros_vec_2d = tl.zeros([BLOCK_SIZE, VEC], dtype=tl.int32)
    s_histogram_rank0_ptr = tle.remote(s_histogram_rank0_ptr, 0, scope=mesh)
    s_step_thresholds_rank0_ptr = tle.remote(s_step_thresholds_rank0_ptr, 0, scope=mesh)
    s_threshold_bin_idx_rank0_ptr = tle.remote(s_threshold_bin_idx_rank0_ptr, 0, scope=mesh)
    s_final_bin_size_rank0_ptr = tle.remote(s_final_bin_size_rank0_ptr, 0, scope=mesh)
    s_out_indices_rank0_ptr = tle.remote(s_out_indices_ptr, 0, scope=mesh)
    s_final_cnt_rank0_ptr = tle.remote(s_final_cnt_ptr, 0, scope=mesh)
    s_found_topk_values_rank0_ptr = tle.remote(s_found_topk_values_ptr, 0, scope=mesh)

    clear_rounds = tl.where(
        step_idx == 3,
        RADIX10_SIZE // BLOCK_SIZE,
        RADIX11_SIZE // BLOCK_SIZE,
    )

    for clear_round in tl.range(0, clear_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_step_local_hist_ptr + clear_bins, 0)
        if is_rank0:
            tl.store(s_histogram_ptr + clear_bins, 0)
    tle.distributed_barrier(mesh)

    if step_idx == 2:
        step1_threshold = tl.load(s_step_thresholds_rank0_ptr + 1)
        logit_pattern = (step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21
    elif step_idx == 3:
        step1_threshold = tl.load(s_step_thresholds_rank0_ptr + 1)
        step2_threshold = tl.load(s_step_thresholds_rank0_ptr + 2)
        logit_pattern = ((step1_threshold.to(tl.uint32) & RADIX11_MASK) << 21) | (
            (step2_threshold.to(tl.uint32) & RADIX11_MASK) << 10)

    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    n_vec_full = seq_len // (BLOCK_SIZE * VEC)
    rem_tiles = (seq_len - n_vec_full * BLOCK_SIZE * VEC) // BLOCK_SIZE

    if ASSUME_ALIGNED:
        for t in tl.range(0, n_vec_full):
            if (t % CLUSTER_SIZE) == cluster_rank:
                base = t * BLOCK_SIZE * VEC + lane * VEC
                offs = base[:, None] + vec[None, :]
                x_vec = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x_vec)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x_vec)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
                elif step_idx == 2:
                    partial = ((key ^ logit_pattern) >> 21) == 0
                else:
                    partial = ((key ^ logit_pattern) >> 10) == 0

                tl.atomic_add(
                    s_step_local_hist_ptr + digit,
                    ones_vec_2d,
                    mask=partial,
                    sem="relaxed",
                    scope="cta",
                )

        for t in tl.range(0, rem_tiles):
            tile_idx = n_vec_full + t
            if (tile_idx % CLUSTER_SIZE) == cluster_rank:
                offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
                x = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = tl.full([BLOCK_SIZE], True, tl.int1)
                elif step_idx == 2:
                    partial = ((key ^ logit_pattern) >> 21) == 0
                else:
                    partial = ((key ^ logit_pattern) >> 10) == 0

                tl.atomic_add(
                    s_step_local_hist_ptr + digit,
                    ones,
                    mask=partial,
                    sem="relaxed",
                    scope="cta",
                )
    else:
        for t in tl.range(0, n_tiles):
            if (t % CLUSTER_SIZE) == cluster_rank:
                offs = t * BLOCK_SIZE + lane
                in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
                key = _convert_to_trt_uint32(x)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = in_range
                elif step_idx == 2:
                    partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
                else:
                    partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

                tl.atomic_add(
                    s_step_local_hist_ptr + digit,
                    ones,
                    mask=partial,
                    sem="relaxed",
                    scope="cta",
                )
    for clear_round in tl.range(0, clear_rounds):
        bins = clear_round * BLOCK_SIZE + lane
        local_counts = tl.load(s_step_local_hist_ptr + bins)
        tl.atomic_add(
            s_histogram_rank0_ptr + bins,
            local_counts,
            sem="relaxed",
            scope="gpu",
        )
    tle.distributed_barrier(mesh)

    found_topk_values = tl.load(s_found_topk_values_rank0_ptr)
    if is_rank0:
        tl.store(s_threshold_bin_idx_ptr, -1)
        tl.store(s_final_bin_size_ptr, 0)
        threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
        final_bin_size_ptrs = s_final_bin_size_ptr + zeros
        last_value = found_topk_values
        threshold_found = False
        for round_idx in tl.range(0, clear_rounds):
            if not threshold_found:
                bins = round_idx * BLOCK_SIZE + lane
                counts = tl.load(s_histogram_ptr + bins)
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
                prefix_sum = prefix_sum + last_value
                total_sum = last_value + counts_total
                next_prefix_sum = prefix_sum + counts
                threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
                threshold_bin = bins
                threshold_bin_size = next_prefix_sum - prefix_sum
                tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
                tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
                found_round = tl.reduce_or(threshold_mask, axis=0)
                threshold_found = found_round
                last_value = total_sum

        threshold_bin_idx_local = tl.load(s_threshold_bin_idx_ptr)
        tl.store(s_step_thresholds_ptr + step_idx, threshold_bin_idx_local)
    tle.distributed_barrier(mesh)

    threshold_bin_idx = tl.load(s_threshold_bin_idx_rank0_ptr)
    final_bin_size = tl.load(s_final_bin_size_rank0_ptr)
    use_final = (step_idx < 3) & (threshold_bin_idx >= 0) & (final_bin_size <= FINAL_SORT_ITEMS)
    if is_rank0 and use_final:
        tl.store(s_final_cnt_ptr, 0)
    tle.distributed_barrier(mesh)

    found_ptrs = s_found_topk_values_rank0_ptr + zeros
    final_cnt_ptrs = s_final_cnt_rank0_ptr + zeros
    if ASSUME_ALIGNED:
        found_ptrs_vec_2d = s_found_topk_values_rank0_ptr + zeros_vec_2d
        final_cnt_ptrs_vec_2d = s_final_cnt_rank0_ptr + zeros_vec_2d
        for t in tl.range(0, n_vec_full):
            if (t % CLUSTER_SIZE) == cluster_rank:
                base = t * BLOCK_SIZE * VEC + lane * VEC
                offs = base[:, None] + vec[None, :]
                x_vec = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x_vec)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x_vec)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = tl.full([BLOCK_SIZE, VEC], True, tl.int1)
                elif step_idx == 2:
                    partial = ((key ^ logit_pattern) >> 21) == 0
                else:
                    partial = ((key ^ logit_pattern) >> 10) == 0

                take_lt = partial & (digit < threshold_bin_idx)
                out_pos_lt = tl.atomic_add(
                    found_ptrs_vec_2d,
                    ones_vec_2d,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_rank0_ptr + out_pos_lt,
                    offs.to(tl.int32),
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if step_idx == 3:
                    take_eq = partial & (digit == threshold_bin_idx)
                    out_pos_eq = tl.atomic_add(
                        s_histogram_rank0_ptr + digit,
                        ones_vec_2d,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_rank0_ptr + out_pos_eq,
                        offs.to(tl.int32),
                        mask=take_eq & (out_pos_eq < TOPK),
                    )
                elif use_final:
                    take_eq_final = partial & (digit == threshold_bin_idx)
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs_vec_2d,
                        ones_vec_2d,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_histogram_rank0_ptr + final_pos,
                        offs.to(tl.int32),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        s_histogram_rank0_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x_vec.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )

        for t in tl.range(0, rem_tiles):
            tile_idx = n_vec_full + t
            if (tile_idx % CLUSTER_SIZE) == cluster_rank:
                offs = (n_vec_full * VEC + t) * BLOCK_SIZE + lane
                x = tl.load(row_ptr + offs)
                key = _convert_to_trt_uint32(x)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = tl.full([BLOCK_SIZE], True, tl.int1)
                elif step_idx == 2:
                    partial = ((key ^ logit_pattern) >> 21) == 0
                else:
                    partial = ((key ^ logit_pattern) >> 10) == 0

                take_lt = partial & (digit < threshold_bin_idx)
                out_pos_lt = tl.atomic_add(
                    found_ptrs,
                    ones,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_rank0_ptr + out_pos_lt,
                    offs.to(tl.int32),
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if step_idx == 3:
                    take_eq = partial & (digit == threshold_bin_idx)
                    out_pos_eq = tl.atomic_add(
                        s_histogram_rank0_ptr + digit,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_rank0_ptr + out_pos_eq,
                        offs.to(tl.int32),
                        mask=take_eq & (out_pos_eq < TOPK),
                    )
                elif use_final:
                    take_eq_final = partial & (digit == threshold_bin_idx)
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs,
                        ones,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_histogram_rank0_ptr + final_pos,
                        offs.to(tl.int32),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        s_histogram_rank0_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
    else:
        for t in tl.range(0, n_tiles):
            if (t % CLUSTER_SIZE) == cluster_rank:
                offs = t * BLOCK_SIZE + lane
                in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
                x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
                key = _convert_to_trt_uint32(x)
                if step_idx == 0:
                    digit = _convert_to_trt_uint16_hi11(x)
                elif step_idx == 1:
                    digit = ((key >> 21) & RADIX11_MASK).to(tl.int32)
                elif step_idx == 2:
                    digit = ((key >> 10) & RADIX11_MASK).to(tl.int32)
                else:
                    digit = (key & RADIX10_MASK).to(tl.int32)

                if step_idx < 2:
                    partial = in_range
                elif step_idx == 2:
                    partial = in_range & (((key ^ logit_pattern) >> 21) == 0)
                else:
                    partial = in_range & (((key ^ logit_pattern) >> 10) == 0)

                take_lt = partial & (digit < threshold_bin_idx)
                out_pos_lt = tl.atomic_add(
                    found_ptrs,
                    ones,
                    mask=take_lt,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    s_out_indices_rank0_ptr + out_pos_lt,
                    offs.to(tl.int32),
                    mask=take_lt & (out_pos_lt < TOPK),
                )

                if step_idx == 3:
                    take_eq = partial & (digit == threshold_bin_idx)
                    out_pos_eq = tl.atomic_add(
                        s_histogram_rank0_ptr + digit,
                        ones,
                        mask=take_eq,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_out_indices_rank0_ptr + out_pos_eq,
                        offs.to(tl.int32),
                        mask=take_eq & (out_pos_eq < TOPK),
                    )
                elif use_final:
                    take_eq_final = partial & (digit == threshold_bin_idx)
                    final_pos = tl.atomic_add(
                        final_cnt_ptrs,
                        ones,
                        mask=take_eq_final,
                        sem="relaxed",
                        scope="cta",
                    )
                    tl.store(
                        s_histogram_rank0_ptr + final_pos,
                        offs.to(tl.int32),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )
                    tl.store(
                        s_histogram_rank0_ptr + (FINAL_SORT_ITEMS + final_pos),
                        x.to(tl.int32, bitcast=True),
                        mask=take_eq_final & (final_pos < FINAL_SORT_ITEMS),
                    )

    tle.distributed_barrier(mesh)

    if step_idx < 3:
        if use_final:
            continue_to_next_step = False
            need_final_sort = True
        else:
            continue_to_next_step = True
            need_final_sort = False
    else:
        if is_rank0:
            tl.store(s_found_topk_values_ptr, TOPK)
        continue_to_next_step = False
        need_final_sort = False

    tle.distributed_barrier(mesh)
    return continue_to_next_step, need_final_sort, logit_pattern


@triton.jit
def tle_topk_selector_kernel_smem_cluster(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    seq_len,
    mesh: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_RADIX_FINAL: tl.constexpr,
):
    cluster_pid = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    pid = cluster_pid // CLUSTER_SIZE
    is_rank0 = cluster_rank == 0

    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)
    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    row_len = row_end - row_start

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    if row_len <= TOPK:
        if is_rank0:
            chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
            for chunk_idx in tl.range(0, chunks):
                pos = chunk_idx * BLOCK_SIZE + lane
                take_row = pos < row_len
                tl.store(out_row + pos * stride_outn, (row_start + pos).to(tl.int32), mask=take_row)
                take_pad = (pos >= row_len) & (pos < TOPK)
                tl.store(out_row + pos * stride_outn, -1, mask=take_pad)
        return

    FINAL_SORT_ITEMS: tl.constexpr = 2048
    HIST_SIZE: tl.constexpr = 4096

    s_histogram = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_step_local_hist = tle.gpu.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPK],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_threshold_bin_idx = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_bin_size = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found_topk_values = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_step_thresholds = tle.gpu.alloc(
        [4],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    s_histogram_ptr = tle.gpu.local_ptr(s_histogram, (0, ))
    s_step_local_hist_ptr = tle.gpu.local_ptr(s_step_local_hist, (0, ))
    s_out_indices_ptr = tle.gpu.local_ptr(s_out_indices, (0, ))
    s_final_cnt_ptr = tle.gpu.local_ptr(s_final_cnt, (0, ))
    s_threshold_bin_idx_ptr = tle.gpu.local_ptr(s_threshold_bin_idx, (0, ))
    s_final_bin_size_ptr = tle.gpu.local_ptr(s_final_bin_size, (0, ))
    s_found_topk_values_ptr = tle.gpu.local_ptr(s_found_topk_values, (0, ))
    s_step_thresholds_ptr = tle.gpu.local_ptr(s_step_thresholds, (0, ))

    s_histogram_rank0 = tle.remote(s_histogram, 0, scope=mesh)
    s_threshold_bin_idx_rank0 = tle.remote(s_threshold_bin_idx, 0, scope=mesh)
    s_final_bin_size_rank0 = tle.remote(s_final_bin_size, 0, scope=mesh)
    s_step_thresholds_rank0 = tle.remote(s_step_thresholds, 0, scope=mesh)

    s_histogram_rank0_ptr = tle.gpu.local_ptr(s_histogram_rank0, (0, ))
    s_threshold_bin_idx_rank0_ptr = tle.gpu.local_ptr(s_threshold_bin_idx_rank0, (0, ))
    s_final_bin_size_rank0_ptr = tle.gpu.local_ptr(s_final_bin_size_rank0, (0, ))
    s_step_thresholds_rank0_ptr = tle.gpu.local_ptr(s_step_thresholds_rank0, (0, ))

    if is_rank0:
        tl.store(s_final_cnt_ptr, 0)
        tl.store(s_threshold_bin_idx_ptr, -1)
        tl.store(s_final_bin_size_ptr, 0)
        tl.store(s_found_topk_values_ptr, 0)
        for i in tl.static_range(4):
            tl.store(s_step_thresholds_ptr + i, 0)
    init_chunks: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
    for init_idx in tl.range(0, init_chunks):
        pos = init_idx * BLOCK_SIZE + lane
        tl.store(s_out_indices_ptr + pos, -1, mask=is_rank0 & (pos < TOPK))
    tle.distributed_barrier(mesh)

    logit_pattern = tl.zeros((), dtype=tl.uint32)
    continue_to_next_step = True
    need_final_sort = False

    for step_idx in tl.static_range(0, 4):
        if continue_to_next_step:
            continue_to_next_step, step_need_final_sort, logit_pattern = _tle_process_histogram_step_cluster(
                row_ptr,
                stride_xn,
                row_start,
                row_end,
                seq_len,
                step_idx,
                logit_pattern,
                cluster_rank,
                is_rank0,
                s_step_local_hist_ptr,
                s_histogram_ptr,
                s_histogram_rank0_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                s_step_thresholds_ptr,
                s_step_thresholds_rank0_ptr,
                s_threshold_bin_idx_ptr,
                s_final_bin_size_ptr,
                s_threshold_bin_idx_rank0_ptr,
                s_final_bin_size_rank0_ptr,
                mesh=mesh,
                CLUSTER_SIZE=CLUSTER_SIZE,
                ASSUME_ALIGNED=ASSUME_ALIGNED,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            need_final_sort = need_final_sort | step_need_final_sort

    if is_rank0 and need_final_sort:
        if USE_RADIX_FINAL:
            _tle_final_select_radix(
                s_histogram_ptr,
                s_out_indices_ptr,
                s_final_cnt_ptr,
                s_found_topk_values_ptr,
                TOPK=TOPK,
                BLOCK_SIZE=BLOCK_SIZE,
                FINAL_SORT_ITEMS=FINAL_SORT_ITEMS,
            )
        else:
            base_idx = tl.load(s_found_topk_values_ptr)
            final_cnt = tl.minimum(tl.load(s_final_cnt_ptr), FINAL_SORT_ITEMS)
            sort_chunks = tl.cdiv(final_cnt, BLOCK_SIZE)
            for sort_chunk in tl.range(0, sort_chunks):
                pos = sort_chunk * BLOCK_SIZE + lane
                valid = pos < final_cnt
                logit_i_bits = tl.load(
                    tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + pos, )),
                    mask=valid,
                    other=0,
                )
                logit_i = logit_i_bits.to(tl.float32, bitcast=True)
                out_rank = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
                for j in tl.range(0, final_cnt):
                    logit_j_bits = tl.load(tle.gpu.local_ptr(s_histogram, (FINAL_SORT_ITEMS + j, )))
                    logit_j = logit_j_bits.to(tl.float32, bitcast=True)
                    better = (logit_i < logit_j) | ((logit_i == logit_j) & (pos < j))
                    out_rank = out_rank + (valid & better).to(tl.int32)
                dst_pos = base_idx + out_rank
                take = valid & (dst_pos < TOPK)
                idx_i = tl.load(
                    tle.gpu.local_ptr(s_histogram, (pos, )),
                    mask=take,
                    other=0,
                )
                tl.store(s_out_indices_ptr + dst_pos, idx_i, mask=take)
            tl.store(s_found_topk_values_ptr, TOPK)
    tle.distributed_barrier(mesh)

    if is_rank0:
        total_out = tl.minimum(tl.load(s_found_topk_values_ptr), TOPK)
        flush_chunks = tl.cdiv(total_out, BLOCK_SIZE)
        for flush_chunk in tl.range(0, flush_chunks):
            pos = flush_chunk * BLOCK_SIZE + lane
            mask = pos < total_out
            out_vals = tl.load(s_out_indices_ptr + pos, mask=mask, other=-1)
            tl.store(out_row + pos * stride_outn, out_vals, mask=mask)
    tle.distributed_barrier(mesh)


@triton.jit
def triton_topk_selector_kernel(
    x_ptr,
    out_ptr,
    cand0_ptr,
    cand1_ptr,
    starts_ptr,
    ends_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_c0m,
    stride_c0n,
    stride_c1m,
    stride_c1n,
    seq_len,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RADIX_BITS: tl.constexpr,
):
    tl.static_assert(RADIX_BITS == 8, "triton_topk_selector_kernel currently expects 8-bit radix")
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)
    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    cand0_row = cand0_ptr + pid * stride_c0m
    cand1_row = cand1_ptr + pid * stride_c1m

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    n_tiles = tl.cdiv(seq_len, BLOCK_SIZE)
    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    bins = tl.arange(0, RADIX_SIZE)

    # Stage 1: 8-bit coarse prescreen on fp16-mapped keys.
    coarse_counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        digit8 = _convert_to_uint16_hi8(x)
        coarse_counts = coarse_counts + tl.histogram(digit8, RADIX_SIZE, mask=in_range)

    coarse_cumsum_desc = tl.cumsum(coarse_counts, axis=0, reverse=True)
    topk_target = TOPK
    coarse_cond = coarse_cumsum_desc > topk_target
    coarse_threshold_bin = tl.max(tl.where(coarse_cond, bins, 0), axis=0).to(tl.int32)
    coarse_counts_gt = tl.max(tl.where(bins == (coarse_threshold_bin + 1), coarse_cumsum_desc, 0), axis=0)
    new_topk = topk_target - coarse_counts_gt
    write_count = 0
    cand_count0 = 0

    # Stage 2: write coarse winners and compact coarse-threshold candidates into cand0.
    for t in tl.range(0, n_tiles):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=float("-inf"))
        digit8 = _convert_to_uint16_hi8(x)
        take = in_range & (digit8 > coarse_threshold_bin)
        take_i32 = take.to(tl.int32)
        pos = write_count + tl.cumsum(take_i32, axis=0) - 1
        mask = take & (pos < TOPK)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=mask)
        write_count = write_count + tl.sum(take_i32, axis=0)

        take_eq = in_range & (digit8 == coarse_threshold_bin)
        take_eq_i32 = take_eq.to(tl.int32)
        pos_eq = cand_count0 + tl.cumsum(take_eq_i32, axis=0) - 1
        tl.store(cand0_row + pos_eq * stride_c0n, offs.to(tl.int32), mask=take_eq)
        cand_count0 = cand_count0 + tl.sum(take_eq_i32, axis=0)

    # Stage 3: four 8-bit refinements over compact candidate lists.
    num_in = cand_count0
    for round_idx in tl.static_range(4):
        if (new_topk > 0) & (num_in > 0):
            shift: tl.constexpr = 24 - round_idx * 8
            desired = tl.zeros((), dtype=tl.uint32)
            desired_mask = tl.zeros((), dtype=tl.uint32)
            radix_mask_u32 = tl.zeros((), dtype=tl.uint32) + RADIX_MASK
            k_to_find = new_topk
            num_in_tiles = tl.cdiv(num_in, BLOCK_SIZE)
            counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)

            # Histogram on current candidate table.
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                if round_idx & 1:
                    idx = tl.load(cand1_row + pos * stride_c1n, mask=valid, other=0)
                else:
                    idx = tl.load(cand0_row + pos * stride_c0n, mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                x_key = _convert_to_uint32(x)
                matches = (x_key & desired_mask) == desired
                take = valid & matches
                digit = ((x_key >> shift) & RADIX_MASK).to(tl.int32)
                counts = counts + tl.histogram(digit, RADIX_SIZE, mask=take)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            cond = cumsum_desc > k_to_find
            threshold_bin = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.max(tl.where(bins == (threshold_bin + 1), cumsum_desc, 0), axis=0)
            desired = desired | (threshold_bin.to(tl.uint32) << shift)
            desired_mask = desired_mask | (radix_mask_u32 << shift)
            new_topk = k_to_find - counts_gt

            out_count = write_count
            next_count = 0
            for t in tl.range(0, num_in_tiles):
                pos = t * BLOCK_SIZE + lane
                valid = pos < num_in
                if round_idx & 1:
                    idx = tl.load(cand1_row + pos * stride_c1n, mask=valid, other=0)
                else:
                    idx = tl.load(cand0_row + pos * stride_c0n, mask=valid, other=0)
                x = tl.load(row_ptr + idx * stride_xn, mask=valid, other=float("-inf"))
                x_key = _convert_to_uint32(x)
                digit = ((x_key >> shift) & RADIX_MASK).to(tl.int32)

                take_gt = valid & (digit > threshold_bin)
                take_gt_i32 = take_gt.to(tl.int32)
                out_pos_gt = out_count + tl.cumsum(take_gt_i32, axis=0) - 1
                out_mask_gt = take_gt & (out_pos_gt < TOPK)
                tl.store(out_row + out_pos_gt * stride_outn, idx, mask=out_mask_gt)
                out_count = out_count + tl.sum(take_gt_i32, axis=0)

                take_eq = valid & (digit == threshold_bin)
                take_eq_i32 = take_eq.to(tl.int32)
                if round_idx == 3:
                    out_pos_eq = out_count + tl.cumsum(take_eq_i32, axis=0) - 1
                    out_mask_eq = take_eq & (out_pos_eq < TOPK)
                    tl.store(out_row + out_pos_eq * stride_outn, idx, mask=out_mask_eq)
                    out_count = out_count + tl.sum(take_eq_i32, axis=0)
                else:
                    nxt_pos = next_count + tl.cumsum(take_eq_i32, axis=0) - 1
                    if round_idx & 1:
                        tl.store(cand0_row + nxt_pos * stride_c0n, idx, mask=take_eq)
                    else:
                        tl.store(cand1_row + nxt_pos * stride_c1n, idx, mask=take_eq)
                    next_count = next_count + tl.sum(take_eq_i32, axis=0)

            write_count = out_count
            num_in = next_count


# %%
# TileLang reference (optional)
# -----------------------------

if _HAVE_TILELANG:
    _TL_PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
    _TL_KERNEL_CACHE = {}

    def convert_to_uint16(x):
        hval = T.Cast(T.float16, x)
        bits_uint = T.reinterpret(T.uint16, hval)
        bits_uint = T.if_then_else(x < 0, ~bits_uint & 0xFFFF, bits_uint | 0x8000)
        return bits_uint >> 8

    def convert_to_uint32(x):
        bits_uint = T.reinterpret(T.uint32, x)
        bits_uint = T.if_then_else(
            x < 0,
            ~bits_uint & T.Cast(T.uint32, 0xFFFFFFFF),
            bits_uint | T.Cast(T.uint32, 0x80000000),
        )
        return bits_uint

    @tilelang.jit(pass_configs=_TL_PASS_CONFIGS)
    def _tilelang_topk_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        RADIX_LOCAL = 1 << 8
        BLOCK_SIZE = 1024
        SMEM_INPUT_SIZE = 4096

        @T.prim_func
        def tl_topk_kernel(
            input: T.Tensor[(batch, seq_len), in_dtype],
            index: T.Tensor[(batch, topk), out_dtype],
            starts: T.Tensor[(batch), out_dtype],
            ends: T.Tensor[(batch), out_dtype],
        ):
            with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
                tx = T.get_thread_binding()

                s_threshold_bin_id = T.alloc_shared([1], T.int32)
                s_histogram = T.alloc_shared([RADIX_LOCAL + 1], T.int32)
                s_num_input = T.alloc_shared([2], T.int32)
                s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

                l_threshold_bin_id = T.alloc_var(T.int32)
                l_new_topk = T.alloc_var(T.int32)
                l_num_input = T.alloc_var(T.int32)
                l_bin_id32 = T.alloc_var(T.int32)
                l_val = T.alloc_var(T.int32)
                l_start_pos = T.alloc_var(T.int32)
                l_start_idx = T.alloc_var(T.int32)
                l_end_idx = T.alloc_var(T.int32)
                l_out_pos = T.alloc_var(T.int32)

                l_new_topk = topk
                l_start_idx = starts[bx]
                l_end_idx = ends[bx]

                T.fill(s_histogram, 0)
                T.fill(s_num_input[0], 0)
                T.sync_threads()
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        inval_int16 = convert_to_uint16(input[bx, input_idx])
                        T.atomic_add(s_histogram[inval_int16], 1)
                T.sync_threads()

                if tx < RADIX_LOCAL:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX_LOCAL)
                        if tx < RADIX_LOCAL - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX_LOCAL)
                        if tx < RADIX_LOCAL - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX_LOCAL)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        bin_id = convert_to_uint16(input[bx, input_idx])
                        l_bin_id32 = T.Cast(T.int32, bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            pos_gt0 = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            index[bx, pos_gt0] = input_idx
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            pos_eq0 = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, pos_eq0] = input_idx

                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()

                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                  (24 - round * 8)) & 0xFF),
                            )
                            T.atomic_add(s_histogram[l_bin_id32], 1)
                    T.sync_threads()

                    if tx < RADIX_LOCAL:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX_LOCAL)
                            if tx < RADIX_LOCAL - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX_LOCAL)
                            if tx < RADIX_LOCAL - offset:
                                s_histogram[tx] = l_val

                        T.sync_threads(3, RADIX_LOCAL)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()

                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        T.sync_threads()
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                  (24 - round * 8)) & 0xFF),
                            )
                            if l_bin_id32 > l_threshold_bin_id:
                                pos_gt_round = T.atomic_add(s_histogram[l_bin_id32 + 1], 1,
                                                            return_prev=True) + l_start_pos
                                index[bx, pos_gt_round] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1,
                                                             return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                else:
                                    pos_eq_round = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, pos_eq_round] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

        return tl_topk_kernel

    def tilelang_topk_selector(input, starts, ends, topk, out: Optional[torch.Tensor] = None):
        batch, _ = input.shape
        if out is None:
            out = torch.zeros((batch, topk), dtype=torch.int32, device=input.device)
        kernel = _TL_KERNEL_CACHE.get(topk)
        if kernel is None:
            kernel = _tilelang_topk_impl(topk)
            _TL_KERNEL_CACHE[topk] = kernel
        kernel(input, out, starts, ends)
        return out


# %%
# Python wrappers
# ---------------


def _supports_tle_cluster_remote() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 9


def tle_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
    use_radix_final: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    tle_block_size = TLE_FIXED_BLOCK_SIZE
    if use_radix_final is None:
        use_radix_final = seq_len >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD

    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % tle_block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    batch, seq_len = x.shape
    grid = (batch, )
    tle_topk_selector_kernel[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=tle_block_size,
        USE_RADIX_FINAL=use_radix_final,
        num_warps=TLE_FIXED_NUM_WARPS,
        num_stages=TLE_FIXED_NUM_STAGES,
    )
    return out


def tle_topk_selector_1024threads(
    x,
    starts,
    ends,
    topk,
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
    use_radix_final: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    tle_block_size = 1024
    if use_radix_final is None:
        use_radix_final = seq_len >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD

    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % tle_block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    grid = (batch, )
    tle_topk_selector_kernel[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=tle_block_size,
        USE_RADIX_FINAL=use_radix_final,
        num_warps=tle_block_size // 32,
        num_stages=TLE_FIXED_NUM_STAGES,
    )
    return out


def tle_topk_selector_smem(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
    use_radix_final: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    tle_block_size = TLE_SMEM_BLOCK_SIZE
    if use_radix_final is None:
        use_radix_final = seq_len >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD
    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % tle_block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    grid = (batch, )
    tle_topk_selector_kernel_smem[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=tle_block_size,
        SMEM_INPUT_SIZE=TLE_SMEM_INPUT_SIZE,
        USE_RADIX_FINAL=use_radix_final,
        num_warps=TLE_SMEM_NUM_WARPS,
        num_stages=TLE_SMEM_NUM_STAGES,
    )
    return out


def tle_topk_selector_smem_cluster(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
    use_radix_final: Optional[bool] = None,
):
    if not _supports_tle_cluster_remote():
        raise RuntimeError("TLE-Cluster requires CUDA SM90+")
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    tle_block_size = TLE_SMEM_BLOCK_SIZE
    if use_radix_final is None:
        use_radix_final = seq_len >= TLE_RADIX_FINAL_SEQ_LEN_THRESHOLD
    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % tle_block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    grid = (batch, )
    tle_topk_selector_kernel_smem_cluster[grid](
        x,
        out,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        seq_len,
        mesh=BLOCK_CLUSTER_MESH_8,
        CLUSTER_SIZE=TLE_SMEM_CLUSTER_SIZE,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=tle_block_size,
        USE_RADIX_FINAL=use_radix_final,
        num_ctas=1,
        num_warps=TLE_SMEM_NUM_WARPS,
        num_stages=TLE_SMEM_NUM_STAGES,
    )
    return out


def triton_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    out: Optional[torch.Tensor] = None,
    cand0: Optional[torch.Tensor] = None,
    cand1: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    if cand0 is None:
        cand0 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)
    if cand1 is None:
        cand1 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)

    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    assert cand0.shape == (batch, seq_len) and cand0.dtype == torch.int32 and cand0.is_cuda
    assert cand1.shape == (batch, seq_len) and cand1.dtype == torch.int32 and cand1.is_cuda

    # Triton kernel uses kernel-specific tuning to avoid slow/unstable configs.
    kernel_num_warps = 4 if block_size >= 1024 else 8

    grid = (batch, )
    triton_topk_selector_kernel[grid](
        x,
        out,
        cand0,
        cand1,
        starts,
        ends,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        cand0.stride(0),
        cand0.stride(1),
        cand1.stride(0),
        cand1.stride(1),
        seq_len,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        RADIX_BITS=8,
        num_warps=kernel_num_warps,
        num_stages=1,
    )
    return out


# %%
# TRT-LLM CUDA reference (optional)
# ---------------------------------

TRTLLM_INDEXER_TOPK_KERNEL_URL = ("https://raw.githubusercontent.com/NVIDIA/TensorRT-LLM/main/"
                                  "cpp/tensorrt_llm/kernels/indexerTopK.cu")
FLASHINFER_TOPK_CUH_URL = ("https://raw.githubusercontent.com/flashinfer-ai/flashinfer/refs/heads/main/"
                           "include/flashinfer/topk.cuh")
FLASHINFER_INCLUDE_BASE_URL = ("https://raw.githubusercontent.com/flashinfer-ai/flashinfer/refs/heads/main/"
                               "include/flashinfer/")
SGLANG_TOPK_KERNEL_URL = ("https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/"
                          "sgl-kernel/csrc/elementwise/topk.cu")

TRTLLM_INDEXER_TOPK_BINDING_CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace kernels {
void invokeIndexerTopKDecode(float const* logits, int const* seqLens, int* indices, float* outLogitsAux,
    int* outIndicesAux, int const splitWorkThreshold, int const numRows, int const numColumns, int const stride0,
    int const stride1, int const next_n, int const topK, cudaStream_t const stream);
void invokeIndexerTopKPrefill(float const* logits, int const* rowStarts, int const* rowEnds, int* indices,
    int const numRows, int const numColumns, int const stride0, int const stride1, int const topK,
    cudaStream_t const stream);
}

void indexer_topk_decode(torch::Tensor logits, torch::Tensor seq_lens, torch::Tensor out,
    torch::Tensor out_logits_aux, torch::Tensor out_indices_aux, int64_t next_n, int64_t topk) {
  TORCH_CHECK(logits.is_cuda() && seq_lens.is_cuda() && out.is_cuda() && out_logits_aux.is_cuda() && out_indices_aux.is_cuda());
  TORCH_CHECK(logits.dtype() == torch::kFloat32);
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32);
  TORCH_CHECK(out.dtype() == torch::kInt32);
  TORCH_CHECK(out_logits_aux.dtype() == torch::kFloat32);
  TORCH_CHECK(out_indices_aux.dtype() == torch::kInt32);
  TORCH_CHECK(logits.dim() == 2 && seq_lens.dim() == 1 && out.dim() == 2);

  int numRows = static_cast<int>(logits.size(0));
  int numColumns = static_cast<int>(logits.size(1));
  TORCH_CHECK(seq_lens.size(0) * static_cast<int>(next_n) == numRows);
  TORCH_CHECK(out.size(0) == numRows && out.size(1) == topk);
  TORCH_CHECK(out_logits_aux.size(0) == numRows && out_indices_aux.size(0) == numRows);
  TORCH_CHECK(out_logits_aux.size(1) >= topk && out_indices_aux.size(1) >= topk);

  kernels::invokeIndexerTopKDecode(
      logits.data_ptr<float>(),
      seq_lens.data_ptr<int>(),
      out.data_ptr<int>(),
      out_logits_aux.data_ptr<float>(),
      out_indices_aux.data_ptr<int>(),
      200 * 1000,
      numRows,
      numColumns,
      static_cast<int>(logits.stride(0)),
      static_cast<int>(logits.stride(1)),
      static_cast<int>(next_n),
      static_cast<int>(topk),
      at::cuda::getDefaultCUDAStream(logits.get_device()));
}

void indexer_topk_prefill(torch::Tensor logits, torch::Tensor row_starts, torch::Tensor row_ends, torch::Tensor out,
    int64_t topk) {
  TORCH_CHECK(logits.is_cuda() && row_starts.is_cuda() && row_ends.is_cuda() && out.is_cuda());
  TORCH_CHECK(logits.dtype() == torch::kFloat32);
  TORCH_CHECK(row_starts.dtype() == torch::kInt32);
  TORCH_CHECK(row_ends.dtype() == torch::kInt32);
  TORCH_CHECK(out.dtype() == torch::kInt32);
  TORCH_CHECK(logits.dim() == 2 && row_starts.dim() == 1 && row_ends.dim() == 1 && out.dim() == 2);

  int numRows = static_cast<int>(logits.size(0));
  int numColumns = static_cast<int>(logits.size(1));
  TORCH_CHECK(row_starts.size(0) == numRows && row_ends.size(0) == numRows);
  TORCH_CHECK(out.size(0) == numRows && out.size(1) == topk);

  kernels::invokeIndexerTopKPrefill(
      logits.data_ptr<float>(),
      row_starts.data_ptr<int>(),
      row_ends.data_ptr<int>(),
      out.data_ptr<int>(),
      numRows,
      numColumns,
      static_cast<int>(logits.stride(0)),
      static_cast<int>(logits.stride(1)),
      static_cast<int>(topk),
      at::cuda::getDefaultCUDAStream(logits.get_device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_topk_decode", &indexer_topk_decode, "TRT-LLM indexerTopK decode");
  m.def("indexer_topk_prefill", &indexer_topk_prefill, "TRT-LLM indexerTopK prefill");
}
"""


def _patch_trtllm_indexer_topk_source(src: str, prefill_threads: int = 512) -> str:
    for old in [
            '#include "moeTopKFuncs.cuh"\n',
            '#include "tensorrt_llm/common/config.h"\n',
            '#include "tensorrt_llm/common/cudaTypeUtils.cuh"\n',
            '#include "tensorrt_llm/common/envUtils.h"\n',
            '#include "tensorrt_llm/kernels/noAuxTcKernels.h"\n',
    ]:
        src = src.replace(old, "")

    if prefill_threads != 512:
        fn_marker = "void invokeIndexerTopKPrefill("
        fn_pos = src.find(fn_marker)
        if fn_pos < 0:
            raise RuntimeError("TRT-LLM source format changed: invokeIndexerTopKPrefill not found")
        tail = src[fn_pos:]
        marker = "constexpr int kNumThreadsPerBlock = 512;"
        rel = tail.find(marker)
        if rel < 0:
            raise RuntimeError("TRT-LLM source format changed: prefill thread marker not found")
        abs_pos = fn_pos + rel
        replacement = f"constexpr int kNumThreadsPerBlock = {prefill_threads};"
        src = src[:abs_pos] + replacement + src[abs_pos + len(marker):]

    # Make the standalone source compile under torch cpp_extension.
    shim = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#define TRTLLM_NAMESPACE_BEGIN
#define TRTLLM_NAMESPACE_END
#define TLLM_CHECK_WITH_INFO(cond, msg) TORCH_CHECK((cond), msg)
namespace tensorrt_llm { namespace common {
inline bool getEnvEnablePDL() { return false; }
inline void sync_check_cuda_error(cudaStream_t) { C10_CUDA_CHECK(cudaGetLastError()); }
}} // namespace tensorrt_llm::common
"""

    return shim + src


@lru_cache(maxsize=4)
def _load_embedded_trtllm_indexer_topk(prefill_threads: int = 512):
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as ex:
        print(f"warning: cannot import torch cpp_extension for trtllm topk: {ex}")
        return None

    try:
        with urllib.request.urlopen(TRTLLM_INDEXER_TOPK_KERNEL_URL, timeout=20) as resp:
            cuda_src = resp.read().decode("utf-8")
    except Exception as ex:
        print(f"warning: failed to download trtllm indexerTopK.cu: {ex}")
        return None

    cuda_src = _patch_trtllm_indexer_topk_source(cuda_src, prefill_threads=prefill_threads)
    digest = hashlib.sha1((TRTLLM_INDEXER_TOPK_BINDING_CPP + cuda_src).encode("utf-8")).hexdigest()[:12]
    ext_name = f"flagtree_trtllm_indexer_topk_{digest}"

    try:
        module = load_inline(
            name=ext_name,
            cpp_sources=[TRTLLM_INDEXER_TOPK_BINDING_CPP],
            cuda_sources=[cuda_src],
            functions=None,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as ex:
        print(f"warning: failed to compile embedded trtllm indexerTopK kernel: {ex}")
        return None

    decode_fn = getattr(module, "indexer_topk_decode", None)
    prefill_fn = getattr(module, "indexer_topk_prefill", None)
    if decode_fn is None:
        print("warning: embedded trtllm topk module has no indexer_topk_decode symbol")
    if prefill_fn is None:
        print("warning: embedded trtllm topk module has no indexer_topk_prefill symbol")
    if decode_fn is None and prefill_fn is None:
        return None
    return module


def trtllm_cuda_topk_selector_decode(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
    out_logits_aux: Optional[torch.Tensor] = None,
    out_indices_aux: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if out is None:
        out = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    # TRT-LLM decode path uses auxiliary buffers in long-sequence branch.
    aux_cols = 10 * topk
    if out_logits_aux is None:
        out_logits_aux = torch.empty((x.shape[0], aux_cols), dtype=torch.float32, device=x.device)
    if out_indices_aux is None:
        out_indices_aux = torch.empty((x.shape[0], aux_cols), dtype=torch.int32, device=x.device)
    module = _load_embedded_trtllm_indexer_topk()
    if module is None:
        raise RuntimeError("TRT-LLM indexerTopK extension unavailable")
    fn = getattr(module, "indexer_topk_decode", None)
    if fn is None:
        raise RuntimeError("TRT-LLM decode symbol unavailable")
    fn(x, ends, out, out_logits_aux, out_indices_aux, 1, int(topk))
    return out


def trtllm_cuda_topk_selector_prefill(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if out is None:
        out = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    module = _load_embedded_trtllm_indexer_topk(prefill_threads=512)
    if module is None:
        raise RuntimeError("TRT-LLM indexerTopK extension unavailable")
    fn = getattr(module, "indexer_topk_prefill", None)
    if fn is None:
        raise RuntimeError("TRT-LLM prefill symbol unavailable")
    fn(x, starts, ends, out, int(topk))
    return out


def trtllm_cuda_topk_selector_prefill_1024threads(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if out is None:
        out = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    module = _load_embedded_trtllm_indexer_topk(prefill_threads=1024)
    if module is None:
        raise RuntimeError("TRT-LLM indexerTopK extension unavailable")
    fn = getattr(module, "indexer_topk_prefill", None)
    if fn is None:
        raise RuntimeError("TRT-LLM prefill symbol unavailable")
    fn(x, starts, ends, out, int(topk))
    return out


FLASHINFER_TOPK_BINDING_CPP = r"""
#include <torch/extension.h>

void flashinfer_topk_cuda(torch::Tensor logits, torch::Tensor out_indices, torch::Tensor out_values,
                          torch::Tensor row_states_buffer, int64_t topk);
int64_t flashinfer_row_state_nbytes();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flashinfer_topk_cuda", &flashinfer_topk_cuda, "FlashInfer TopKDispatch CUDA");
  m.def("flashinfer_row_state_nbytes", &flashinfer_row_state_nbytes, "FlashInfer RadixRowState bytes");
}
"""

FLASHINFER_TOPK_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include "flashinfer/math.cuh"
#include "flashinfer/topk.cuh"

void flashinfer_topk_cuda(torch::Tensor logits, torch::Tensor out_indices, torch::Tensor out_values,
                          torch::Tensor row_states_buffer, int64_t topk) {
  TORCH_CHECK(logits.is_cuda() && out_indices.is_cuda() && out_values.is_cuda() && row_states_buffer.is_cuda());
  TORCH_CHECK(logits.dtype() == torch::kFloat32);
  TORCH_CHECK(out_indices.dtype() == torch::kInt32);
  TORCH_CHECK(out_values.dtype() == torch::kFloat32);
  TORCH_CHECK(row_states_buffer.dtype() == torch::kUInt8);
  TORCH_CHECK(logits.dim() == 2 && out_indices.dim() == 2 && out_values.dim() == 2);

  int num_rows = static_cast<int>(logits.size(0));
  int max_len = static_cast<int>(logits.size(1));
  TORCH_CHECK(out_indices.size(0) == num_rows && out_indices.size(1) == topk);
  TORCH_CHECK(out_values.size(0) == num_rows && out_values.size(1) == topk);
  TORCH_CHECK(topk > 0 && topk <= max_len);

  const int64_t need_nbytes = static_cast<int64_t>(num_rows) *
      static_cast<int64_t>(sizeof(flashinfer::sampling::RadixRowState));
  TORCH_CHECK(row_states_buffer.numel() >= need_nbytes);

  auto* row_states = reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states_buffer.data_ptr<uint8_t>());
  auto err = flashinfer::sampling::TopKDispatch<float, int32_t>(
      logits.data_ptr<float>(),
      out_indices.data_ptr<int32_t>(),
      out_values.data_ptr<float>(),
      static_cast<uint32_t>(num_rows),
      static_cast<uint32_t>(topk),
      static_cast<uint32_t>(max_len),
      row_states,
      at::cuda::getDefaultCUDAStream(logits.get_device()));
  C10_CUDA_CHECK(err);
}

int64_t flashinfer_row_state_nbytes() {
  return static_cast<int64_t>(sizeof(flashinfer::sampling::RadixRowState));
}
"""


@lru_cache(maxsize=1)
def _prepare_embedded_flashinfer_headers():
    import re
    import tempfile
    from pathlib import Path

    include_root = Path(tempfile.gettempdir()) / "flagtree_flashinfer_include_main"
    header_root = include_root / "flashinfer"
    header_root.mkdir(parents=True, exist_ok=True)

    queue = ["topk.cuh", "math.cuh"]
    seen = set()
    while queue:
        header = queue.pop(0)
        if header in seen:
            continue
        seen.add(header)
        local_path = header_root / header
        if local_path.exists():
            src = local_path.read_text(encoding="utf-8")
        else:
            header_url = FLASHINFER_TOPK_CUH_URL if header == "topk.cuh" else (FLASHINFER_INCLUDE_BASE_URL + header)
            with urllib.request.urlopen(header_url, timeout=20) as resp:
                src = resp.read().decode("utf-8")
            local_path.write_text(src, encoding="utf-8")
        for inc in re.findall(r'^#include\s+"([^"]+)"', src, flags=re.M):
            if inc not in seen:
                queue.append(inc)
    return str(include_root)


@lru_cache(maxsize=1)
def _load_embedded_flashinfer_topk():
    try:
        from pathlib import Path
        from torch.utils.cpp_extension import load_inline
    except Exception as ex:
        print(f"warning: cannot import torch cpp_extension for flashinfer topk: {ex}")
        return None

    try:
        include_dir = _prepare_embedded_flashinfer_headers()
        topk_src = (Path(include_dir) / "flashinfer" / "topk.cuh").read_text(encoding="utf-8")
    except Exception as ex:
        print(f"warning: failed to prepare flashinfer headers: {ex}")
        return None

    digest = hashlib.sha1(
        (FLASHINFER_TOPK_BINDING_CPP + FLASHINFER_TOPK_CUDA_SRC + topk_src).encode("utf-8")).hexdigest()[:12]
    ext_name = f"flagtree_flashinfer_topk_{digest}"

    try:
        module = load_inline(
            name=ext_name,
            cpp_sources=[FLASHINFER_TOPK_BINDING_CPP],
            cuda_sources=[FLASHINFER_TOPK_CUDA_SRC],
            functions=None,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
            ],
            with_cuda=True,
            extra_include_paths=[include_dir],
            verbose=False,
        )
    except Exception as ex:
        print(f"warning: failed to compile embedded flashinfer topk: {ex}")
        return None

    if getattr(module, "flashinfer_topk_cuda", None) is None:
        print("warning: embedded flashinfer module has no flashinfer_topk_cuda symbol")
        return None
    if getattr(module, "flashinfer_row_state_nbytes", None) is None:
        print("warning: embedded flashinfer module has no flashinfer_row_state_nbytes symbol")
        return None
    return module


def flashinfer_cuda_topk_selector(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
    out_values: Optional[torch.Tensor] = None,
    row_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if x.dtype != torch.float32:
        x = x.float()
    if out is None:
        out = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if out_values is None:
        out_values = torch.empty((x.shape[0], topk), dtype=torch.float32, device=x.device)
    module = _load_embedded_flashinfer_topk()
    if module is None:
        raise RuntimeError("FlashInfer topk extension unavailable")
    if row_states is None:
        row_state_nbytes = int(module.flashinfer_row_state_nbytes())
        row_states = torch.zeros((x.shape[0] * row_state_nbytes, ), dtype=torch.uint8, device=x.device)
    # Current benchmark path uses starts==0 and ends==seq_len; decode/ragged transforms are not used.
    module.flashinfer_topk_cuda(x, out, out_values, row_states, int(topk))
    return out


SGLANG_TOPK_BINDING_CPP = r"""
#include <torch/extension.h>
#include <optional>

void fast_topk_interface(
    const at::Tensor& score,
    at::Tensor& indices,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt);

void sglang_fast_topk(torch::Tensor score, torch::Tensor lengths, torch::Tensor out) {
  TORCH_CHECK(score.is_cuda() && lengths.is_cuda() && out.is_cuda());
  TORCH_CHECK(score.dtype() == torch::kFloat32);
  TORCH_CHECK(lengths.dtype() == torch::kInt32);
  TORCH_CHECK(out.dtype() == torch::kInt32);
  TORCH_CHECK(score.dim() == 2 && lengths.dim() == 1 && out.dim() == 2);
  TORCH_CHECK(score.size(0) == lengths.size(0));
  TORCH_CHECK(score.size(0) == out.size(0));
  TORCH_CHECK(out.size(1) == 2048, "sglang fast_topk kernel is fixed TopK=2048");
  fast_topk_interface(score, out, lengths, std::nullopt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sglang_fast_topk", &sglang_fast_topk, "SGLang fast topk");
}
"""


@lru_cache(maxsize=1)
def _load_embedded_sglang_topk():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as ex:
        print(f"warning: cannot import torch cpp_extension for sglang topk: {ex}")
        return None

    try:
        with urllib.request.urlopen(SGLANG_TOPK_KERNEL_URL, timeout=20) as resp:
            cuda_src = resp.read().decode("utf-8")
    except Exception as ex:
        print(f"warning: failed to download sglang topk.cu: {ex}")
        return None

    digest = hashlib.sha1((SGLANG_TOPK_BINDING_CPP + cuda_src).encode("utf-8")).hexdigest()[:12]
    ext_name = f"flagtree_sglang_topk_{digest}"
    try:
        module = load_inline(
            name=ext_name,
            cpp_sources=[SGLANG_TOPK_BINDING_CPP],
            cuda_sources=[cuda_src],
            functions=None,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    except Exception as ex:
        print(f"warning: failed to compile embedded sglang topk kernel: {ex}")
        return None

    fn = getattr(module, "sglang_fast_topk", None)
    if fn is None:
        print("warning: embedded sglang topk module has no sglang_fast_topk symbol")
    return fn


def sglang_cuda_topk_selector(
    x: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if topk != 2048:
        raise RuntimeError("sglang fast_topk kernel only supports topk=2048")
    if x.dtype != torch.float32:
        x = x.float()
    if out is None:
        out = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    lengths = (ends - starts).to(torch.int32)
    fn = _load_embedded_sglang_topk()
    if fn is None:
        raise RuntimeError("SGLang topk extension unavailable")
    fn(x, lengths, out)
    return out


# %%
# Correctness & benchmarking
# --------------------------


def _torch_topk_indices(x, starts, ends, topk):
    batch, _ = x.shape
    out = torch.empty((batch, topk), dtype=torch.int32, device=x.device)
    for i in range(batch):
        start = int(starts[i].item())
        end = int(ends[i].item())
        vals, idx = torch.topk(x[i, start:end], topk, dim=0)
        out[i] = idx.to(torch.int32) + start
    return out


def _recall(pred, ref):
    batch = pred.shape[0]
    k = ref.shape[1]
    hits = 0
    for i in range(batch):
        pred_set = set(pred[i].tolist())
        ref_set = set(ref[i].tolist())
        hits += len(pred_set & ref_set)
    return hits / (batch * k)


_BENCH_PROVIDERS = (["triton"] + ["trtllm-prefill"] + ["trtllm-prefill-1024threads"] + ["flashinfer-cuda"] +
                    ["tle-trt"] + ["tle-trt-1024threads"] + ["tle-cluster"] + (["tilelang"] if _HAVE_TILELANG else []))
_BENCH_NAMES = (["Triton"] + ["TRTLLM-Prefill"] + ["TRTLLM-Prefill-1024T"] + ["FlashInfer"] + ["TLE-TRT"] +
                ["TLE-TRT-1024T"] + ["TLE-Cluster"] + (["TileLang"] if _HAVE_TILELANG else []))
_BENCH_STYLES = ([("red", "-")] + [("black", "-")] + [("brown", "-")] + [("gray", "-")] + [("orange", "-")] +
                 [("olive", "-")] + [("teal", "-")] + ([("blue", "-")] if _HAVE_TILELANG else []))
_BENCH_XVALS = [
    (1, 131072, 2048),
    (1, 262144, 2048),
    (1, 524288, 2048),
    (64, 4096, 128),
    (64, 8192, 256),
    (64, 32768, 1024),
    (64, 32768, 2048),
    (64, 131072, 2048),
    (64, 524288, 2048),
]
_TILELANG_SKIP_SEQ_LEN_MIN = 262144


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch", "seq_len", "topk"],
        x_vals=_BENCH_XVALS,
        x_log=True,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="topk-selector",
        args={},
    ))
def benchmark(batch, seq_len, topk, provider, block_size, warmup, rep):
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "tle-trt":
        tle_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            tle_topk_selector(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
                out=tle_out,
                assume_aligned=assume_aligned,
            )

    elif provider == "tle-trt-1024threads":
        tle_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            tle_topk_selector_1024threads(
                x,
                starts,
                ends,
                topk,
                out=tle_out,
                assume_aligned=assume_aligned,
            )

    elif provider == "tle-cluster":
        if not _supports_tle_cluster_remote():
            return float("nan"), float("nan"), float("nan")
        if batch >= 48 and seq_len >= 131072:
            return float("nan"), float("nan"), float("nan")
        tle_smem_cluster_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            tle_topk_selector_smem_cluster(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
                out=tle_smem_cluster_out,
                assume_aligned=assume_aligned,
            )

    elif provider == "triton":
        triton_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
        triton_cand0 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)
        triton_cand1 = torch.empty((batch, seq_len), dtype=torch.int32, device=x.device)

        def run():
            triton_topk_selector(
                x,
                starts,
                ends,
                topk,
                block_size=block_size,
                out=triton_out,
                cand0=triton_cand0,
                cand1=triton_cand1,
                assume_aligned=assume_aligned,
            )

    elif provider == "trtllm-decode":
        if _load_embedded_trtllm_indexer_topk() is None:
            return float("nan"), float("nan"), float("nan")
        trtllm_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
        trtllm_out_logits_aux = torch.empty((batch, 10 * topk), dtype=torch.float32, device=x.device)
        trtllm_out_indices_aux = torch.empty((batch, 10 * topk), dtype=torch.int32, device=x.device)

        def run():
            trtllm_cuda_topk_selector_decode(
                x,
                starts,
                ends,
                topk,
                out=trtllm_out,
                out_logits_aux=trtllm_out_logits_aux,
                out_indices_aux=trtllm_out_indices_aux,
            )

    elif provider == "trtllm-prefill":
        if _load_embedded_trtllm_indexer_topk() is None:
            return float("nan"), float("nan"), float("nan")
        trtllm_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            trtllm_cuda_topk_selector_prefill(
                x,
                starts,
                ends,
                topk,
                out=trtllm_out,
            )

    elif provider == "trtllm-prefill-1024threads":
        if _load_embedded_trtllm_indexer_topk(prefill_threads=1024) is None:
            return float("nan"), float("nan"), float("nan")
        trtllm_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            trtllm_cuda_topk_selector_prefill_1024threads(
                x,
                starts,
                ends,
                topk,
                out=trtllm_out,
            )

    elif provider == "flashinfer-cuda":
        module = _load_embedded_flashinfer_topk()
        if module is None:
            return float("nan"), float("nan"), float("nan")
        flashinfer_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
        flashinfer_out_values = torch.empty((batch, topk), dtype=torch.float32, device=x.device)
        row_state_nbytes = int(module.flashinfer_row_state_nbytes())
        flashinfer_row_states = torch.zeros((batch * row_state_nbytes, ), dtype=torch.uint8, device=x.device)

        def run():
            flashinfer_cuda_topk_selector(
                x,
                starts,
                ends,
                topk,
                out=flashinfer_out,
                out_values=flashinfer_out_values,
                row_states=flashinfer_row_states,
            )

    elif provider == "sglang-cuda":
        if topk != 2048:
            return float("nan"), float("nan"), float("nan")
        if _load_embedded_sglang_topk() is None:
            return float("nan"), float("nan"), float("nan")
        sglang_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

        def run():
            sglang_cuda_topk_selector(x, starts, ends, topk, out=sglang_out)

    elif provider == "torch":

        def run():
            torch.topk(x, topk, dim=-1)[1]

    else:
        if not _HAVE_TILELANG:
            return float("nan"), float("nan"), float("nan")
        if seq_len >= _TILELANG_SKIP_SEQ_LEN_MIN:
            return float("nan"), float("nan"), float("nan")
        tilelang_out = torch.zeros((batch, topk), dtype=torch.int32, device=x.device)

        def run():
            tilelang_topk_selector(x, starts, ends, topk, out=tilelang_out)

    ms, min_ms, max_ms = triton.testing.do_bench(
        run,
        quantiles=quantiles,
        warmup=warmup,
        rep=rep,
    )
    return ms, max_ms, min_ms


def run_correctness(batch, seq_len, topk, block_size):
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)

    ref = _torch_topk_indices(x, starts, ends, topk)

    tle_out = tle_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        assume_aligned=assume_aligned,
    )
    tle_smem_out = tle_topk_selector_smem(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        assume_aligned=assume_aligned,
    )
    tle_smem_cluster_out = None
    if _supports_tle_cluster_remote():
        tle_smem_cluster_out = tle_topk_selector_smem_cluster(
            x,
            starts,
            ends,
            topk,
            block_size=block_size,
            assume_aligned=assume_aligned,
        )

    print(f"TLE recall vs torch.topk: {_recall(tle_out, ref):.4f}")
    print(f"TLE-SMEM recall vs torch.topk: {_recall(tle_smem_out, ref):.4f}")
    if tle_smem_cluster_out is not None:
        print(f"TLE-Cluster recall vs torch.topk: {_recall(tle_smem_cluster_out, ref):.4f}")
    else:
        print("TLE-Cluster not available; skipping cluster correctness.")
    triton_out = triton_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        assume_aligned=assume_aligned,
    )
    print(f"Triton recall vs torch.topk: {_recall(triton_out, ref):.4f}")
    print(f"TLE recall vs Triton: {_recall(tle_out, triton_out):.4f}")
    print(f"TLE-SMEM recall vs Triton: {_recall(tle_smem_out, triton_out):.4f}")
    if tle_smem_cluster_out is not None:
        print(f"TLE-Cluster recall vs Triton: {_recall(tle_smem_cluster_out, triton_out):.4f}")

    trtllm_fn = _load_embedded_trtllm_indexer_topk()
    if trtllm_fn is not None:
        trtllm_out = trtllm_cuda_topk_selector_decode(x, starts, ends, topk)
        print(f"TRTLLM-CUDA-Decode recall vs torch.topk: {_recall(trtllm_out, ref):.4f}")
        print(f"TRTLLM-CUDA-Decode recall vs Triton: {_recall(trtllm_out, triton_out):.4f}")
    else:
        print("TRTLLM-CUDA-Decode not available; skipping TRTLLM correctness.")

    flashinfer_mod = _load_embedded_flashinfer_topk()
    if flashinfer_mod is not None:
        flashinfer_out = flashinfer_cuda_topk_selector(x, starts, ends, topk)
        print(f"FlashInfer-CUDA recall vs torch.topk: {_recall(flashinfer_out, ref):.4f}")
        print(f"FlashInfer-CUDA recall vs Triton: {_recall(flashinfer_out, triton_out):.4f}")
    else:
        print("FlashInfer-CUDA not available; skipping FlashInfer correctness.")

    sglang_fn = _load_embedded_sglang_topk()
    if topk == 2048 and sglang_fn is not None:
        sglang_out = sglang_cuda_topk_selector(x, starts, ends, topk)
        print(f"SGLang-CUDA recall vs torch.topk: {_recall(sglang_out, ref):.4f}")
        print(f"SGLang-CUDA recall vs Triton: {_recall(sglang_out, triton_out):.4f}")
    elif topk != 2048:
        print("SGLang-CUDA only supports topk=2048; skipping SGLang correctness.")
    else:
        print("SGLang-CUDA not available; skipping SGLang correctness.")

    if _HAVE_TILELANG:
        tilelang_out = tilelang_topk_selector(x, starts, ends, topk)
        print(f"TileLang recall vs torch.topk: {_recall(tilelang_out, ref):.4f}")
        print(f"TLE recall vs TileLang: {_recall(tle_out, tilelang_out):.4f}")
    else:
        print("TileLang not available; skipping TileLang correctness.")


def _parse_bench_x_vals(raw):
    if not raw:
        return None
    vals = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        parts = text.split("x")
        if len(parts) != 3:
            raise ValueError(f"invalid --bench_x_vals item: {text!r}, expect BxSxK")
        vals.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not vals:
        raise ValueError("--bench_x_vals produced empty set")
    return vals


def _parse_providers(raw):
    if not raw:
        return None
    providers = [p.strip() for p in raw.split(",") if p.strip()]
    if not providers:
        raise ValueError("--providers produced empty set")
    providers = ["trtllm-decode" if p == "trtllm-cuda" else p for p in providers]
    unknown = [p for p in providers if p not in _BENCH_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown providers: {unknown}, supported={list(_BENCH_PROVIDERS)}")
    return providers


def run_bench(block_size, warmup, rep, show_plots, providers=None, bench_x_vals=None, quick_bench=False,
              max_seq_len=None):
    bench = benchmark.benchmarks

    x_vals = list(_BENCH_XVALS)
    if quick_bench:
        x_vals = [v for v in x_vals if v[1] <= 32768]
    if max_seq_len is not None:
        x_vals = [v for v in x_vals if v[1] <= max_seq_len]
    if bench_x_vals is not None:
        x_vals = list(bench_x_vals)
    if not x_vals:
        raise ValueError("no benchmark x_vals left after filtering")

    line_vals = list(_BENCH_PROVIDERS)
    line_names = list(_BENCH_NAMES)
    styles = list(_BENCH_STYLES)
    if providers is not None:
        index_map = {p: i for i, p in enumerate(_BENCH_PROVIDERS)}
        selected_indices = [index_map[p] for p in providers]
        line_vals = [line_vals[i] for i in selected_indices]
        line_names = [line_names[i] for i in selected_indices]
        styles = [styles[i] for i in selected_indices]

    bench.x_vals = x_vals
    bench.line_vals = line_vals
    bench.line_names = line_names
    bench.styles = styles
    print(f"[bench] providers={line_vals}, x_vals={x_vals}, warmup={warmup}, rep={rep}, block_size={block_size}")

    benchmark.run(
        print_data=True,
        show_plots=show_plots,
        block_size=block_size,
        warmup=warmup,
        rep=rep,
    )


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--topk", type=int, default=128, help="top-k")
    parser.add_argument("--block_size", type=int, default=1024, help="block size (threads)")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iters")
    parser.add_argument("--rep", type=int, default=20, help="benchmark iters")
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness check")
    parser.add_argument("--skip_bench", action="store_true", help="skip benchmark")
    parser.add_argument(
        "--providers",
        type=str,
        default="",
        help=(
            "comma-separated providers for benchmark, e.g. "
            "tle-trt,tle-trt-1024threads,tle-cluster,triton,trtllm-prefill,trtllm-prefill-1024threads,flashinfer-cuda"),
    )
    parser.add_argument(
        "--bench_x_vals",
        type=str,
        default="",
        help="override x-values: comma-separated BxSxK triplets, e.g. 64x4096x128,64x8192x256",
    )
    parser.add_argument(
        "--quick_bench",
        action="store_true",
        help="benchmark only default cases with seq_len<=32768",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="filter benchmark cases by seq_len <= max_seq_len",
    )
    args = parser.parse_args(argv)

    if not args.skip_correctness:
        run_correctness(
            batch=args.batch,
            seq_len=args.seq_len,
            topk=args.topk,
            block_size=args.block_size,
        )

    if not args.skip_bench:
        providers = _parse_providers(args.providers)
        bench_x_vals = _parse_bench_x_vals(args.bench_x_vals)
        run_bench(
            block_size=args.block_size,
            warmup=args.warmup,
            rep=args.rep,
            show_plots=args.show_plots,
            providers=providers,
            bench_x_vals=bench_x_vals,
            quick_bench=args.quick_bench,
            max_seq_len=args.max_seq_len,
        )


if __name__ == "__main__":
    main()
