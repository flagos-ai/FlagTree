"""
DeepSeek V3-2 Top-K Selector with Triton and TLE (TLE Tutorial)
==============================================================

This tutorial adapts the TileLang DeepSeek V3-2 top-k selector example and
implements two kernels:
- A Triton version that uses global scratch buffers for histograms and
  candidate queues.
- A TLE version that uses shared memory (`tle.alloc` + `tle.local_ptr`).

Both kernels implement the same radix-selection flow as the TileLang reference:
1) Stage-1: bucket by the top 8 bits of FP16 keys to find a coarse threshold.
2) Stage-2: refine ties using 4 rounds of 8-bit radix passes on FP32 keys.

If TileLang is installed, the script will also run the original TileLang kernel
and compare correctness and performance.

Notes
-----
- Input dtype is assumed to be float32 for the 32-bit radix refinement.
- `SMEM_INPUT_SIZE` bounds the number of candidates carried into stage-2.
  If the threshold bucket exceeds this size, results are approximate.
"""

# %%
# Setup
# -----

import argparse
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle

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

# %%
# Key conversions
# ---------------


@triton.jit
def _convert_to_uint16(x):
    hval = x.to(tl.float16)
    bits = hval.to(tl.uint16, bitcast=True)
    sign_mask = tl.full(hval.shape, 0x8000, tl.uint16)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits >> 8


@triton.jit
def _convert_to_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(bits.shape, 0x80000000, tl.uint32)
    bits = tl.where(x < 0, ~bits, bits | sign_mask)
    return bits


# %%
# Triton kernel (global scratch)
# ------------------------------


@triton.jit
def triton_topk_selector_kernel(
    x_ptr,
    out_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    num_ptr,
    input_idx_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    stride_hist,
    stride_num,
    stride_input_m,
    stride_input_r,
    stride_input_c,
    seq_len,
    RADIX: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    SMEM_INPUT: tl.constexpr,
    NUM_INPUT_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm
    hist_row = hist_ptr + pid * stride_hist
    num_row = num_ptr + pid * stride_num
    input_row = input_idx_ptr + pid * stride_input_m

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)

    hist_idx = tl.arange(0, RADIX)
    hist_last = tl.full([1], RADIX, tl.int32)

    # init histogram + counters
    tl.store(hist_row + hist_idx, 0)
    tl.store(hist_row + hist_last, 0)
    tl.store(num_row + tl.arange(0, 2), 0)
    tl.debug_barrier()

    l_new_topk = tl.full((), TOPK, tl.int32)

    # stage 1: 8-bit FP16 bucket
    for t in tl.static_range(N_TILES):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        tl.atomic_add(hist_row + bin_i32, ones, mask=in_range)

    # reverse cumsum on histogram[0:RADIX)
    rev_idx = (RADIX - 1) - hist_idx
    hist_rev = tl.load(hist_row + rev_idx)
    hist_cum_rev = tl.cumsum(hist_rev, axis=0)
    tl.store(hist_row + rev_idx, hist_cum_rev)
    tl.debug_barrier()

    hist_cum = tl.load(hist_row + hist_idx)
    hist_cum_next = tl.load(hist_row + hist_idx + 1, mask=hist_idx + 1 < RADIX, other=0)
    cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
    cand = tl.where(cond, hist_idx.to(tl.int32), -1)
    threshold = tl.max(cand, axis=0)
    hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
    l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

    # collect bins > threshold into output, == threshold into candidate buffer
    num_ptrs = num_row + tl.zeros([BLOCK_SIZE], tl.int32)
    for t in tl.static_range(N_TILES):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        gt_thr = bin_i32 > threshold
        eq_thr = bin_i32 == threshold

        pos = tl.atomic_add(hist_row + bin_i32 + 1, ones, mask=in_range & gt_thr)
        pos = tl.where(in_range & gt_thr, pos, 0)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=in_range & gt_thr & (pos < TOPK))

        pos_eq = tl.atomic_add(num_ptrs, ones, mask=in_range & eq_thr & (l_new_topk > 0))
        pos_eq = tl.where(in_range & eq_thr, pos_eq, 0)
        tl.store(
            input_row + pos_eq * stride_input_c,
            offs.to(tl.int32),
            mask=in_range & eq_thr & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
        )

    # stage 2: 4 rounds of 8-bit radix on FP32 keys
    for round_id in tl.static_range(4):
        r_idx = round_id & 1
        next_idx = r_idx ^ 1
        start_pos = TOPK - l_new_topk

        # reset histogram and next round counter
        tl.store(hist_row + hist_idx, 0)
        tl.store(hist_row + hist_last, 0)
        tl.store(num_row + next_idx, 0)
        tl.debug_barrier()

        l_num_input = tl.load(num_row + r_idx).to(tl.int32)
        max_input = tl.full((), SMEM_INPUT, tl.int32)
        l_num_input = tl.minimum(l_num_input, max_input)
        active = l_new_topk > 0

        shift = 24 - round_id * 8
        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                input_row + r_idx * stride_input_r + offs * stride_input_c,
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)
            tl.atomic_add(hist_row + bin_i32, ones, mask=valid & active)

        rev_idx = (RADIX - 1) - hist_idx
        hist_rev = tl.load(hist_row + rev_idx)
        hist_cum_rev = tl.cumsum(hist_rev, axis=0)
        tl.store(hist_row + rev_idx, hist_cum_rev)
        tl.debug_barrier()

        hist_cum = tl.load(hist_row + hist_idx)
        hist_cum_next = tl.load(hist_row + hist_idx + 1, mask=hist_idx + 1 < RADIX, other=0)
        cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
        cand = tl.where(cond, hist_idx.to(tl.int32), -1)
        threshold = tl.max(cand, axis=0)
        hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
        l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                input_row + r_idx * stride_input_r + offs * stride_input_c,
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)

            gt_thr = bin_i32 > threshold
            eq_thr = bin_i32 == threshold
            pos = tl.atomic_add(hist_row + bin_i32 + 1, ones, mask=valid & gt_thr & active)
            pos = tl.where(valid & gt_thr & active, pos, 0)
            out_pos = pos + start_pos
            tl.store(
                out_row + out_pos * stride_outn,
                cand_idx,
                mask=valid & gt_thr & active & (out_pos < TOPK),
            )

            if round_id == 3:
                pos_eq = tl.atomic_add(hist_row + bin_i32 + 1, ones, mask=valid & eq_thr & active & (l_new_topk > 0))
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                out_pos = pos_eq + start_pos
                tl.store(
                    out_row + out_pos * stride_outn,
                    cand_idx,
                    mask=valid & eq_thr & active & (out_pos < TOPK) & (l_new_topk > 0),
                )
            else:
                num_ptrs = num_row + next_idx + tl.zeros([BLOCK_SIZE], tl.int32)
                pos_eq = tl.atomic_add(num_ptrs, ones, mask=valid & eq_thr & active & (l_new_topk > 0))
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                tl.store(
                    input_row + next_idx * stride_input_r + pos_eq * stride_input_c,
                    cand_idx,
                    mask=valid & eq_thr & active & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
                )


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
    RADIX: tl.constexpr,
    HIST_SIZE: tl.constexpr,
    ASSUME_ALIGNED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_TILES: tl.constexpr,
    SMEM_INPUT: tl.constexpr,
    NUM_INPUT_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = tl.load(starts_ptr + pid).to(tl.int32)
    row_end = tl.load(ends_ptr + pid).to(tl.int32)

    row_ptr = x_ptr + pid * stride_xm
    out_row = out_ptr + pid * stride_outm

    if ASSUME_ALIGNED:
        tl.assume(row_start == 0)
        tl.assume(row_end == seq_len)
        tl.assume(stride_xn == 1)
        tl.assume(stride_outn == 1)
        seq_len = tl.multiple_of(seq_len, BLOCK_SIZE)

    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.int32)

    s_histogram = tle.alloc(
        [HIST_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    s_num_input = tle.alloc(
        [2],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    s_input_idx = tle.alloc(
        [2, SMEM_INPUT],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )

    hist_idx = tl.arange(0, RADIX)
    hist_last = tl.full([1], RADIX, tl.int32)

    hist_ptrs = tle.local_ptr(s_histogram, (hist_idx, ))
    hist_last_ptrs = tle.local_ptr(s_histogram, (hist_last, ))
    tl.store(hist_ptrs, 0)
    tl.store(hist_last_ptrs, 0)
    tl.store(tle.local_ptr(s_num_input, (tl.arange(0, 2), )), 0)
    tl.debug_barrier()

    l_new_topk = tl.full((), TOPK, tl.int32)

    # stage 1
    for t in tl.static_range(N_TILES):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        hist_bin_ptrs = tle.local_ptr(s_histogram, (bin_i32, ))
        tl.atomic_add(hist_bin_ptrs, ones, mask=in_range)

    rev_idx = (RADIX - 1) - hist_idx
    hist_rev = tl.load(tle.local_ptr(s_histogram, (rev_idx, )))
    hist_cum_rev = tl.cumsum(hist_rev, axis=0)
    tl.store(tle.local_ptr(s_histogram, (rev_idx, )), hist_cum_rev)
    tl.debug_barrier()

    hist_cum = tl.load(hist_ptrs)
    hist_cum_next = tl.load(tle.local_ptr(s_histogram, (hist_idx + 1, )), mask=hist_idx + 1 < RADIX, other=0)
    cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
    cand = tl.where(cond, hist_idx.to(tl.int32), -1)
    threshold = tl.max(cand, axis=0)
    hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
    l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

    num_ptrs = tle.local_ptr(s_num_input, (tl.zeros([BLOCK_SIZE], tl.int32), ))
    for t in tl.static_range(N_TILES):
        offs = t * BLOCK_SIZE + lane
        in_range = (offs < seq_len) & (offs >= row_start) & (offs < row_end)
        x = tl.load(row_ptr + offs * stride_xn, mask=in_range, other=0.0)
        bin_u16 = _convert_to_uint16(x)
        bin_i32 = bin_u16.to(tl.int32)
        gt_thr = bin_i32 > threshold
        eq_thr = bin_i32 == threshold

        pos = tl.atomic_add(tle.local_ptr(s_histogram, (bin_i32 + 1, )), ones, mask=in_range & gt_thr)
        pos = tl.where(in_range & gt_thr, pos, 0)
        tl.store(out_row + pos * stride_outn, offs.to(tl.int32), mask=in_range & gt_thr & (pos < TOPK))

        pos_eq = tl.atomic_add(num_ptrs, ones, mask=in_range & eq_thr & (l_new_topk > 0))
        pos_eq = tl.where(in_range & eq_thr, pos_eq, 0)
        tl.store(
            tle.local_ptr(s_input_idx, (tl.zeros([BLOCK_SIZE], tl.int32), pos_eq)),
            offs.to(tl.int32),
            mask=in_range & eq_thr & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
        )

    # stage 2
    for round_id in tl.static_range(4):
        r_idx = round_id & 1
        next_idx = r_idx ^ 1
        start_pos = TOPK - l_new_topk

        tl.store(hist_ptrs, 0)
        tl.store(hist_last_ptrs, 0)
        num_ptrs_next = tle.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], next_idx, tl.int32), ))
        tl.store(num_ptrs_next, 0, mask=lane == 0)
        tl.debug_barrier()

        num_ptrs_r = tle.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], r_idx, tl.int32), ))
        l_num_input = tl.max(tl.load(num_ptrs_r), axis=0).to(tl.int32)
        max_input = tl.full((), SMEM_INPUT, tl.int32)
        l_num_input = tl.minimum(l_num_input, max_input)
        active = l_new_topk > 0

        shift = 24 - round_id * 8
        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                tle.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs)),
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)
            tl.atomic_add(tle.local_ptr(s_histogram, (bin_i32, )), ones, mask=valid & active)

        rev_idx = (RADIX - 1) - hist_idx
        hist_rev = tl.load(tle.local_ptr(s_histogram, (rev_idx, )))
        hist_cum_rev = tl.cumsum(hist_rev, axis=0)
        tl.store(tle.local_ptr(s_histogram, (rev_idx, )), hist_cum_rev)
        tl.debug_barrier()

        hist_cum = tl.load(hist_ptrs)
        hist_cum_next = tl.load(tle.local_ptr(s_histogram, (hist_idx + 1, )), mask=hist_idx + 1 < RADIX, other=0)
        cond = (hist_cum > l_new_topk) & (hist_cum_next <= l_new_topk)
        cand = tl.where(cond, hist_idx.to(tl.int32), -1)
        threshold = tl.max(cand, axis=0)
        hist_next = tl.max(tl.where(hist_idx == threshold + 1, hist_cum, 0), axis=0)
        l_new_topk = tl.maximum(l_new_topk - hist_next, 0)

        for t in tl.static_range(NUM_INPUT_TILES):
            offs = t * BLOCK_SIZE + lane
            valid = offs < l_num_input
            cand_idx = tl.load(
                tle.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], r_idx, tl.int32), offs)),
                mask=valid,
                other=0,
            )
            x = tl.load(row_ptr + cand_idx * stride_xn, mask=valid, other=0.0)
            bin_u32 = _convert_to_uint32(x)
            bin_i32 = ((bin_u32 >> shift) & 0xFF).to(tl.int32)

            gt_thr = bin_i32 > threshold
            eq_thr = bin_i32 == threshold
            pos = tl.atomic_add(tle.local_ptr(s_histogram, (bin_i32 + 1, )), ones, mask=valid & gt_thr & active)
            pos = tl.where(valid & gt_thr & active, pos, 0)
            out_pos = pos + start_pos
            tl.store(
                out_row + out_pos * stride_outn,
                cand_idx,
                mask=valid & gt_thr & active & (out_pos < TOPK),
            )

            if round_id == 3:
                pos_eq = tl.atomic_add(
                    tle.local_ptr(s_histogram, (bin_i32 + 1, )),
                    ones,
                    mask=valid & eq_thr & active & (l_new_topk > 0),
                )
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                out_pos = pos_eq + start_pos
                tl.store(
                    out_row + out_pos * stride_outn,
                    cand_idx,
                    mask=valid & eq_thr & active & (out_pos < TOPK) & (l_new_topk > 0),
                )
            else:
                num_ptrs = tle.local_ptr(s_num_input, (tl.full([BLOCK_SIZE], next_idx, tl.int32), ))
                pos_eq = tl.atomic_add(num_ptrs, ones, mask=valid & eq_thr & active & (l_new_topk > 0))
                pos_eq = tl.where(valid & eq_thr & active, pos_eq, 0)
                tl.store(
                    tle.local_ptr(s_input_idx, (tl.full([BLOCK_SIZE], next_idx, tl.int32), pos_eq)),
                    cand_idx,
                    mask=valid & eq_thr & active & (pos_eq < SMEM_INPUT) & (l_new_topk > 0),
                )


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
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            index[bx, pos] = input_idx
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, pos] = input_idx

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
                                pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1,
                                                             return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                else:
                                    pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

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


def _allocate_triton_scratch(batch, smem_input, device):
    hist = torch.empty((batch, RADIX + 1), dtype=torch.int32, device=device)
    num = torch.empty((batch, 2), dtype=torch.int32, device=device)
    input_idx = torch.empty((batch, 2, smem_input), dtype=torch.int32, device=device)
    return hist, num, input_idx


def triton_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    num_warps=32,
    smem_input=4096,
    out: Optional[torch.Tensor] = None,
    scratch=None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    if scratch is None:
        scratch = _allocate_triton_scratch(batch, smem_input, x.device)
    hist, num, input_idx = scratch

    n_tiles = triton.cdiv(seq_len, block_size)
    num_input_tiles = triton.cdiv(smem_input, block_size)
    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % block_size == 0)
                          and torch.all(starts == 0).item() and torch.all(ends == seq_len).item())

    grid = (batch, )
    triton_topk_selector_kernel[grid](
        x,
        out,
        starts,
        ends,
        hist,
        num,
        input_idx,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        hist.stride(0),
        num.stride(0),
        input_idx.stride(0),
        input_idx.stride(1),
        input_idx.stride(2),
        seq_len,
        RADIX=RADIX,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        N_TILES=n_tiles,
        SMEM_INPUT=smem_input,
        NUM_INPUT_TILES=num_input_tiles,
        num_warps=num_warps,
    )
    return out


def tle_topk_selector(
    x,
    starts,
    ends,
    topk,
    block_size=1024,
    num_warps=32,
    smem_input=4096,
    out: Optional[torch.Tensor] = None,
    assume_aligned: Optional[bool] = None,
):
    if x.dtype != torch.float32:
        x = x.float()
    batch, seq_len = x.shape
    if out is None:
        out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

    n_tiles = triton.cdiv(seq_len, block_size)
    num_input_tiles = triton.cdiv(smem_input, block_size)
    hist_size = RADIX * 2
    if assume_aligned is None:
        assume_aligned = (x.is_contiguous() and out.is_contiguous() and (seq_len % block_size == 0)
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
        RADIX=RADIX,
        HIST_SIZE=hist_size,
        ASSUME_ALIGNED=assume_aligned,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        N_TILES=n_tiles,
        SMEM_INPUT=smem_input,
        NUM_INPUT_TILES=num_input_tiles,
        num_warps=num_warps,
    )
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


def _bench(label, fn, warmup=5, rep=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / rep
    print(f"{label}: {ms:.3f} ms")
    return ms


def run_correctness(batch, seq_len, topk, block_size, smem_input, num_warps):
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)

    ref = _torch_topk_indices(x, starts, ends, topk)

    triton_out = triton_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        num_warps=num_warps,
        smem_input=smem_input,
        assume_aligned=assume_aligned,
    )
    tle_out = tle_topk_selector(
        x,
        starts,
        ends,
        topk,
        block_size=block_size,
        num_warps=num_warps,
        smem_input=smem_input,
        assume_aligned=assume_aligned,
    )

    print(f"Triton recall vs torch.topk: {_recall(triton_out, ref):.4f}")
    print(f"TLE recall vs torch.topk: {_recall(tle_out, ref):.4f}")

    if _HAVE_TILELANG:
        tilelang_out = tilelang_topk_selector(x, starts, ends, topk)
        print(f"TileLang recall vs torch.topk: {_recall(tilelang_out, ref):.4f}")
        print(f"Triton recall vs TileLang: {_recall(triton_out, tilelang_out):.4f}")
        print(f"TLE recall vs TileLang: {_recall(tle_out, tilelang_out):.4f}")
    else:
        print("TileLang not available; skipping TileLang correctness.")


def run_bench(batch, seq_len, topk, block_size, smem_input, num_warps, warmup, rep):
    torch.manual_seed(1)
    x = torch.randn(batch, seq_len, device=DEVICE, dtype=torch.float32)
    starts = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ends = torch.full((batch, ), seq_len, dtype=torch.int32, device=DEVICE)
    assume_aligned = (seq_len % block_size == 0)

    triton_scratch = _allocate_triton_scratch(batch, smem_input, x.device)
    triton_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)
    tle_out = torch.full((batch, topk), -1, dtype=torch.int32, device=x.device)

    def run_triton():
        triton_topk_selector(
            x,
            starts,
            ends,
            topk,
            block_size=block_size,
            num_warps=num_warps,
            smem_input=smem_input,
            out=triton_out,
            scratch=triton_scratch,
            assume_aligned=assume_aligned,
        )

    def run_tle():
        tle_topk_selector(
            x,
            starts,
            ends,
            topk,
            block_size=block_size,
            num_warps=num_warps,
            smem_input=smem_input,
            out=tle_out,
            assume_aligned=assume_aligned,
        )

    def run_torch():
        torch.topk(x, topk, dim=-1)[1]

    print("Benchmark (ms):")
    _bench("Triton", run_triton, warmup=warmup, rep=rep)
    _bench("TLE", run_tle, warmup=warmup, rep=rep)
    _bench("torch.topk", run_torch, warmup=warmup, rep=rep)

    if _HAVE_TILELANG:
        tilelang_out = torch.zeros((batch, topk), dtype=torch.int32, device=x.device)

        def run_tilelang():
            tilelang_topk_selector(x, starts, ends, topk, out=tilelang_out)

        _bench("TileLang", run_tilelang, warmup=warmup, rep=rep)
    else:
        print("TileLang not available; skipping TileLang benchmark.")


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--topk", type=int, default=128, help="top-k")
    parser.add_argument("--block_size", type=int, default=1024, help="block size (threads)")
    parser.add_argument("--smem_input", type=int, default=4096, help="candidate buffer size")
    parser.add_argument("--num_warps", type=int, default=32, help="num warps")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iters")
    parser.add_argument("--rep", type=int, default=20, help="benchmark iters")
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness check")
    parser.add_argument("--skip_bench", action="store_true", help="skip benchmark")
    args = parser.parse_args(argv)

    if args.topk > args.smem_input:
        raise ValueError("topk must be <= smem_input to avoid truncation")

    if not args.skip_correctness:
        run_correctness(
            batch=args.batch,
            seq_len=args.seq_len,
            topk=args.topk,
            block_size=args.block_size,
            smem_input=args.smem_input,
            num_warps=args.num_warps,
        )

    if not args.skip_bench:
        run_bench(
            batch=args.batch,
            seq_len=args.seq_len,
            topk=args.topk,
            block_size=args.block_size,
            smem_input=args.smem_input,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
        )


if __name__ == "__main__":
    main()
