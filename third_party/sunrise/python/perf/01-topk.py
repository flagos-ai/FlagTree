"""
Top-K with Triton (Radix-Only Tutorial)
=======================================

This tutorial implements Top-K over the last dimension of an (M, N) tensor and
compares:
- radix: Triton radix-select kernel
- triton: Triton streaming top-k kernel
- torch: torch.topk
"""

import argparse
import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x_bits):
    tm, fm = get_topmask_and_fullmask(x_bits)
    mask = tl.where((x_bits & tm) != 0, fm, tm)
    return x_bits ^ mask


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    mask = tl.where((x & tm) != 0, tm, fm)
    return x ^ mask


@triton.jit
def indx_to_key(indx):
    max_u16 = tl.full(indx.shape, 0xFFFF, dtype=tl.uint32)
    return max_u16 - indx.to(tl.uint32)


@triton.jit
def key_to_indx(indx_key):
    max_u16 = tl.full(indx_key.shape, 0xFFFF, dtype=tl.uint32)
    return (max_u16 - indx_key.to(tl.uint32)).to(tl.int32)


@triton.jit
def topk_kernel_radix_triton(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    CACHE_TILES0: tl.constexpr,
    CACHE_TILES1: tl.constexpr,
    KEYS_ALLOC0: tl.constexpr,
    KEYS_ALLOC1: tl.constexpr,
    KOUT_ALLOC: tl.constexpr,
):
    pid = tl.program_id(0)
    # Stage 0: setup dtype metadata.
    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
    x_utype = tl.dtype(f"uint{x_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    bins = tl.arange(0, RADIX_SIZE)
    one = tl.full([BLOCK_N], 1, tl.int32)
    # Total cached tiles across both buffers; tiles at or beyond this are re-read
    # from global memory on every pass.
    CACHE_TILES: tl.constexpr = CACHE_TILES0 + CACHE_TILES1

    desired = tl.full((), 0, dtype=x_utype)
    desired_mask = tl.full((), 0, dtype=x_utype)
    k_to_find = tl.full((), K, dtype=tl.int32)
    k_limit = tl.full((), K, dtype=tl.int32)
    n_tiles = tl.cdiv(n_cols, BLOCK_N)

    # Stage 1: shared-memory histogram storage for each radix digit.
    smem_counts = tle.gpu.alloc(
        [RADIX_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_count_ptrs = tle.gpu.local_ptr(smem_counts, (bins, ))

    # Stage 1.5 (shared-memory key cache, TWO buffers): read the first
    # CACHE_TILES tiles of the row from global memory ONCE into shared memory as
    # order-preserving keys. Because fpval_to_key is monotonic, every later pass
    # reads cached keys (in key space) instead of re-fetching X. Tiles at or
    # beyond CACHE_TILES (the tail that does not fit the 128 KB budget) are
    # re-read from global on each pass.
    #
    # tle.gpu.alloc requires a power-of-two shape, and a single power-of-two
    # buffer caps usable coverage at 64 KB (the next-lower power of two below the
    # 128 KB budget once the histogram is subtracted). Splitting the cache across
    # TWO power-of-two buffers lets us pack ~96 KB (e.g. 16+8 tiles) and reach
    # ~75% coverage. buffer 0 holds tiles [0, CACHE_TILES0); buffer 1 holds tiles
    # [CACHE_TILES0, CACHE_TILES). Loops use tl.range (NOT static_range) so they
    # are not unrolled — compile time stays low.
    smem_keys0 = tle.gpu.alloc(
        [KEYS_ALLOC0],
        dtype=x_utype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_keys1 = tle.gpu.alloc(
        [KEYS_ALLOC1],
        dtype=x_utype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    for t in tl.range(0, CACHE_TILES0):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        x = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=float("-inf"))
        # Padding lanes (offs_n >= n_cols) become the key of -inf (the minimum),
        # so they never match a selected bucket and never pass a threshold.
        tl.store(tle.gpu.local_ptr(smem_keys0, (offs_n, )), fpval_to_key(x.to(x_utype, bitcast=True)))
    for t in tl.range(0, CACHE_TILES1):
        offs_g = (CACHE_TILES0 + t) * BLOCK_N + tl.arange(0, BLOCK_N)  # global column
        offs_l = t * BLOCK_N + tl.arange(0, BLOCK_N)  # local index in buffer 1
        mask_n = offs_g < n_cols
        x = tl.load(X + pid * stride_xm + offs_g, mask=mask_n, other=float("-inf"))
        tl.store(tle.gpu.local_ptr(smem_keys1, (offs_l, )), fpval_to_key(x.to(x_utype, bitcast=True)))

    # Stage 2: MSD radix-select; pick one digit bucket per pass. Cached tiles
    # read keys from shared memory (buffer 0 or 1); the tail is re-read from global.
    for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
        tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            if t < CACHE_TILES0:
                x_key = tl.load(tle.gpu.local_ptr(smem_keys0, (offs_n, )))
            elif t < CACHE_TILES:
                offs_l = offs_n - CACHE_TILES0 * BLOCK_N
                x_key = tl.load(tle.gpu.local_ptr(smem_keys1, (offs_l, )))
            else:
                x = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=float("-inf"))
                x_key = fpval_to_key(x.to(x_utype, bitcast=True))
            matches = (x_key & desired_mask) == desired
            digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
            valid = mask_n & matches
            count_addrs = tle.gpu.local_ptr(smem_counts, (digit, ))
            tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

        counts = tl.load(smem_count_ptrs)

        # Compute descending cumulative histogram in-place.
        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)

        cond = cumsum_desc >= k_to_find
        selected = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
        counts_gt = tl.max(tl.where(bins == (selected + 1), cumsum_desc, 0), axis=0)

        selected_u = selected.to(x_utype)
        desired = desired | (selected_u << digit_pos)
        desired_mask = desired_mask | (tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos)
        k_to_find = k_to_find - counts_gt

    # Stage 3: compact the top-K candidates, then write them out COALESCED.
    #
    # The candidates are scattered (their column positions are data-dependent),
    # so we first stage them into shared memory at the slot returned by an
    # atomic counter — a shared scatter costs ~19 cycles vs a global scatter's
    # thousands (uncoalesced, non-contiguous global writes serialize on the
    # STCU write interface, which showed up as core_wr_average_latency ~8000).
    # After both selection passes fill the shared buffers, a single contiguous
    # sweep writes exactly K elements to global — fully coalesced, like the
    # streaming kernel's final store. All comparisons stay in key space
    # (fpval_to_key is monotonic); the float value is recovered on write-out.
    thr_key = desired

    smem_write_count = tle.gpu.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # Shared staging buffers for the K output values (as float bits) and indices.
    # KOUT_ALLOC is next_pow2(K) so the alloc shape is a power of two.
    smem_out_val = tle.gpu.alloc(
        [KOUT_ALLOC],
        dtype=x_dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_out_idx = tle.gpu.alloc(
        [KOUT_ALLOC],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle.gpu.local_ptr(smem_write_count, (0, )), 0)
    write_count_ptrs = tle.gpu.local_ptr(smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32), ))

    # Pass 1: stage all values strictly greater than threshold (in key space)
    # into the shared output buffers at their atomic slot.
    for t in tl.range(0, n_tiles):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        if t < CACHE_TILES0:
            x_key = tl.load(tle.gpu.local_ptr(smem_keys0, (offs_n, )))
        elif t < CACHE_TILES:
            offs_l = offs_n - CACHE_TILES0 * BLOCK_N
            x_key = tl.load(tle.gpu.local_ptr(smem_keys1, (offs_l, )))
        else:
            x = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=float("-inf"))
            x_key = fpval_to_key(x.to(x_utype, bitcast=True))
        take_gt = mask_n & (x_key > thr_key)
        pos = tl.atomic_add(write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta")
        write_mask = take_gt & (pos < k_limit)
        out_pos = pos.to(tl.int32)
        x_val = key_to_fpval(x_key).to(x_dtype, bitcast=True)
        tl.store(tle.gpu.local_ptr(smem_out_val, (out_pos, )), x_val, mask=write_mask)
        tl.store(tle.gpu.local_ptr(smem_out_idx, (out_pos, )), offs_n.to(tl.int32), mask=write_mask)

    # Pass 2: fill remaining slots with values equal to threshold (first-come-first-serve).
    cur_count = tl.load(tle.gpu.local_ptr(smem_write_count, (0, )))
    if cur_count < k_limit:
        for t in tl.range(0, n_tiles):
            cur_count = tl.load(tle.gpu.local_ptr(smem_write_count, (0, )))
            if cur_count < k_limit:
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                if t < CACHE_TILES0:
                    x_key = tl.load(tle.gpu.local_ptr(smem_keys0, (offs_n, )))
                elif t < CACHE_TILES:
                    offs_l = offs_n - CACHE_TILES0 * BLOCK_N
                    x_key = tl.load(tle.gpu.local_ptr(smem_keys1, (offs_l, )))
                else:
                    x = tl.load(X + pid * stride_xm + offs_n, mask=mask_n, other=float("-inf"))
                    x_key = fpval_to_key(x.to(x_utype, bitcast=True))
                take_eq = mask_n & (x_key == thr_key)
                pos = tl.atomic_add(write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta")
                write_mask = take_eq & (pos < k_limit)
                out_pos = pos.to(tl.int32)
                x_val = key_to_fpval(x_key).to(x_dtype, bitcast=True)
                tl.store(tle.gpu.local_ptr(smem_out_val, (out_pos, )), x_val, mask=write_mask)
                tl.store(tle.gpu.local_ptr(smem_out_idx, (out_pos, )), offs_n.to(tl.int32), mask=write_mask)

    # Final: one contiguous, fully-coalesced write of exactly K elements to
    # global memory (read back from the shared staging buffers).
    offs_k = tl.arange(0, K)
    out_vals = tl.load(tle.gpu.local_ptr(smem_out_val, (offs_k, )))
    out_idxs = tl.load(tle.gpu.local_ptr(smem_out_idx, (offs_k, )))
    tl.store(Yv + pid * stride_ym + offs_k, out_vals)
    tl.store(Yi + pid * stride_ym + offs_k, out_idxs)


@triton.jit
def topk_kernel_streaming_triton(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    x_dtype: tl.constexpr = X.dtype.element_ty
    x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
    x_utype = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        packed_nbits: tl.constexpr = 32
    else:
        packed_nbits: tl.constexpr = x_nbits * 2
    x_packtype = tl.dtype(f"uint{packed_nbits}")

    n_tiles = tl.cdiv(n_cols, BLOCK_N)
    offs_n = (n_tiles - 1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_cols

    x_ptrs = X + pid * stride_xm + offs_n
    x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
    x_key = fpval_to_key(x.to(x_utype, bitcast=True))
    x_pack = (x_key.to(x_packtype) << 16) | indx_to_key(offs_n).to(x_packtype)
    acc = tl.topk(x_pack, K)

    for _ in tl.range(0, n_tiles - 1):
        acc = tl.bitonic_merge(acc)
        offs_n -= BLOCK_N
        x_ptrs = X + pid * stride_xm + offs_n
        x = tl.load(x_ptrs, mask=tl.full([BLOCK_N], True, tl.int1), other=float("-inf"))
        x_key = fpval_to_key(x.to(x_utype, bitcast=True))
        x_pack = (x_key.to(x_packtype) << 16) | indx_to_key(offs_n).to(x_packtype)
        acc = tl.maximum(acc, tl.topk(x_pack, K))

    # Rotate index-key into high bits, then sort by descending key.
    acc = (acc << (packed_nbits - 16)) | (acc >> 16)
    acc = tl.sort(acc, descending=True)

    y_indx_key = (acc >> (packed_nbits - 16)).to(tl.uint32)
    y_idx = key_to_indx(y_indx_key)
    y_val_bits = acc.to(x_utype)
    y_vals = key_to_fpval(y_val_bits).to(x_dtype, bitcast=True)

    offs_k = tl.arange(0, K)
    tl.store(Yv + pid * stride_ym + offs_k, y_vals)
    tl.store(Yi + pid * stride_ym + offs_k, y_idx)


def triton_radix_topk(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
):
    assert x.device.type == DEVICE.type, "input must be on device"
    assert x.ndim == 2, "input must be 2D (M, N)"
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")

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

    num_batch = n_rows
    num_blocks = num_batch
    # Tuned heuristic from empirical sweeps:
    # - medium/large N prefers BLOCK_N=1024 and higher warp count
    # - very small N should avoid over-large BLOCK_N
    block_n_radix = max(32, triton.next_power_of_2(n_cols))
    block_n_radix = min(block_n_radix, 1024)
    if block_n_radix <= 64:
        num_warps = 2
    elif block_n_radix <= 128:
        num_warps = 4
    else:
        num_warps = 8
    # Shared-memory key cache across TWO power-of-two buffers. A single
    # power-of-two allocation caps usable coverage at 64 KB for N=32768 (the
    # next-lower power of two below the 128 KB budget once the histogram is
    # subtracted). Two buffers let us pack more: we take as many tiles as the
    # budget allows, then split that count into two power-of-two chunks
    # (largest-first), e.g. 24 tiles -> 16 + 8 => 96 KB, ~75% coverage. Tiles
    # beyond CACHE_TILES0+CACHE_TILES1 fall back to global re-reads each pass.
    RADIX_BITS = 8
    SMEM_BUDGET_BYTES = 128 * 1024
    SMEM_MARGIN_BYTES = 4 * 1024
    other_smem = (1 << RADIX_BITS) * 4 + 4  # histogram (int32) + write counter
    key_budget = SMEM_BUDGET_BYTES - SMEM_MARGIN_BYTES - other_smem
    # Cached keys are the unsigned integer of the same bitwidth as the input
    # (uint16 for fp16, uint32 for fp32), so size the cache by the actual key
    # byte width — NOT a hardcoded 4 — otherwise fp16 wastes half the SMEM budget.
    key_bytes = x.element_size()
    n_tiles_radix = triton.cdiv(n_cols, block_n_radix)
    max_cache_tiles = key_budget // (block_n_radix * key_bytes)
    cache_tiles = min(n_tiles_radix, max_cache_tiles)
    assert cache_tiles >= 1, "shared-memory budget too small to cache even one tile"

    def _floor_pow2(v):
        return triton.next_power_of_2(v + 1) // 2 if v >= 1 else 0

    # Largest power-of-two chunk first, then the largest power-of-two of the
    # remainder. Both chunks are powers of two and BLOCK_N is a power of two, so
    # each buffer's allocation (chunk * BLOCK_N) is a valid power-of-two shape.
    cache_tiles0 = _floor_pow2(cache_tiles)
    cache_tiles1 = _floor_pow2(cache_tiles - cache_tiles0)
    keys_alloc0 = cache_tiles0 * block_n_radix
    keys_alloc1 = max(1, cache_tiles1 * block_n_radix)  # alloc must be non-empty
    # Shared staging for the K outputs; power-of-two shape for tle.gpu.alloc.
    kout_alloc = triton.next_power_of_2(k)
    topk_kernel_radix_triton[(num_blocks, )](
        x,
        y_vals,
        y_idx,
        x.stride(0),
        y_vals.stride(0),
        n_cols,
        K=k,
        BLOCK_N=block_n_radix,
        RADIX_BITS=RADIX_BITS,
        CACHE_TILES0=cache_tiles0,
        CACHE_TILES1=cache_tiles1,
        KEYS_ALLOC0=keys_alloc0,
        KEYS_ALLOC1=keys_alloc1,
        KOUT_ALLOC=kout_alloc,
        num_warps=num_warps,
        num_stages=1,
    )
    return y_vals, y_idx


def triton_topk(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
):
    assert x.device.type == DEVICE.type, "input must be on device"
    assert x.ndim == 2, "input must be 2D (M, N)"
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")
    if n_cols > 65535:
        raise ValueError(f"triton_topk supports N <= 65535, got N={n_cols}")

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

    block_n = max(32, triton.next_power_of_2(min(n_cols, 1024)))
    if block_n <= 64:
        num_warps = 2
    elif block_n <= 128:
        num_warps = 4
    else:
        num_warps = 8

    topk_kernel_streaming_triton[(n_rows, )](
        x,
        y_vals,
        y_idx,
        x.stride(0),
        y_vals.stride(0),
        n_cols,
        K=k,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    return y_vals, y_idx


def _get_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def run_correctness(m: int, n: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.rand((m, n), device=DEVICE, dtype=dtype)
    t_vals, _ = torch.topk(x, k, dim=1, sorted=False)
    t_vals_sorted = torch.sort(t_vals, dim=1, descending=True).values
    t_vals_sorted = t_vals_sorted.cpu()

    y_vals_radix, y_idx_radix = triton_radix_topk(x, k)
    y_vals_radix = y_vals_radix.cpu()
    y_idx_radix = y_idx_radix.cpu()
    y_vals_radix_sorted = torch.sort(y_vals_radix, dim=1, descending=True).values
    torch.testing.assert_close(y_vals_radix_sorted, t_vals_sorted, rtol=1e-3, atol=1e-3)
    gathered_radix = x.cpu().gather(1, y_idx_radix.to(torch.int64))
    torch.testing.assert_close(gathered_radix, y_vals_radix, rtol=1e-3, atol=1e-3)

    y_vals_triton, y_idx_triton = triton_topk(x, k)
    y_vals_triton = y_vals_triton.cpu()
    y_idx_triton = y_idx_triton.cpu()
    y_vals_triton_sorted = torch.sort(y_vals_triton, dim=1, descending=True).values
    torch.testing.assert_close(y_vals_triton_sorted, t_vals_sorted, rtol=1e-3, atol=1e-3)
    gathered_triton = x.cpu().gather(1, y_idx_triton.to(torch.int64))
    torch.testing.assert_close(gathered_triton, y_vals_triton, rtol=1e-3, atol=1e-3)

    print("Correctness check passed (radix + triton).")


if "--only_unit_test" in sys.argv:
    _args = argparse.Namespace(batch=8, seq_len=128, K=2, dtype="float16")
    _dtype = _get_dtype(_args.dtype)
    run_correctness(_args.batch, _args.seq_len, _args.K, _dtype)
    sys.exit(0)

_BENCH_PROVIDERS = ["radix", "triton", "torch"]
_BENCH_NAMES = ["Triton-RadixSelect", "Triton-TopK", "Torch-TopK"]
_BENCH_STYLES = [("red", "-"), ("blue", "-"), ("orange", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[
            (64, 128, 8),
            (64, 1024, 32),
            (64, 8192, 128),
            (128, 32768, 256),
        ],
        x_log=True,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS[:-1],
        line_names=_BENCH_NAMES[:-1],
        styles=_BENCH_STYLES[:-1],
        ylabel="ms",
        plot_name="tle-topk-radix-vs-triton-vs-torch",
        args={},
    ))
def benchmark(M, N, K, provider, dtype):
    bench_warmup = 1
    bench_rep = 3
    N = int(N)
    x = torch.rand((M, N), device=DEVICE, dtype=dtype)
    y_vals = torch.empty((M, K), device=DEVICE, dtype=dtype)
    y_idx = torch.empty((M, K), device=DEVICE, dtype=torch.int32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "radix":

        def run_kernel():
            triton_radix_topk(x, K, out_vals=y_vals, out_idx=y_idx)

        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    elif provider == "triton":

        def run_kernel():
            triton_topk(x, K, out_vals=y_vals, out_idx=y_idx)

        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    else:

        def run_kernel():
            torch.topk(x, K, dim=1, sorted=False)

        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    return ms, max_ms, min_ms


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="sequence length")
    parser.add_argument("--K", type=int, default=2, help="topk")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--skip_correctness", action="store_true", help="skip correctness check before benchmark")
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)
    check_m = args.batch
    check_n = min(args.seq_len, 256)
    check_k = min(args.K, check_n)
    if not args.skip_correctness:
        run_correctness(check_m, check_n, check_k, dtype)

    benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype)


if __name__ == "__main__":
    main()
