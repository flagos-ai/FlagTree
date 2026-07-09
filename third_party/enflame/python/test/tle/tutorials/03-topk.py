"""
Top-K with Triton on Enflame GCU (Radix-Select Tutorial)
=========================================================

This tutorial implements Top-K over the last dimension of an (M, N) tensor
on Enflame GCU and compares:
- naive: Triton radix-select kernel (pure Triton, no TLE, tiled)
- tle:   TLE-optimized radix-select kernel (SMEM L1 cache, full-row CTA)
- torch: torch.topk

Both kernels use MSD (Most Significant Digit) radix-select with RADIX_BITS=5
(32 bins, 7 rounds for fp32), persistent grid=(24,), num_warps=4 to align
with topsop's CTA configuration.

The TLE variant caches the entire row in L1/VDMEM (shared memory) via
tle.gpu.alloc, reading from global memory only once. All radix rounds and
the reclaim phase operate on SMEM, giving a significant bandwidth advantage
for large N.

Algorithm:
  Phase 1 — MSD radix-select: for each of 7 radix rounds, compute histogram
            (tl.histogram → MRADIX HW), reverse cumsum (tl.cumsum → miota HW),
            select the digit bucket containing the K-th element.
  Phase 2 — Reclaim: gather elements > threshold (gt pass) then == threshold
            (eq pass) into the output using cumsum-based compaction.
"""

import argparse
import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ---------------------------------------------------------------------------
# Shared helpers: float ↔ radix key conversion (sign-magnitude → ordered uint)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Kernel 1: Naive radix-select (pure Triton, tiled, no TLE)
# ---------------------------------------------------------------------------


@triton.jit
def topk_kernel_naive(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_rows,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    NUM_DIGITS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
    x_utype = tl.dtype(f"uint{x_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS  # type: ignore[assignment, operator]
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1  # type: ignore[assignment]
    bins = tl.arange(0, RADIX_SIZE)
    START_POS: tl.constexpr = (NUM_DIGITS - 1) * RADIX_BITS  # type: ignore[operator]

    for row in tl.range(pid, n_rows, num_pids):
        base = row * stride_xm
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        k_limit = tl.full((), K, dtype=tl.int32)

        for digit_pos in tl.static_range(START_POS, -1, -RADIX_BITS):
            counts = tl.zeros([RADIX_SIZE], dtype=tl.int32)
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                x = tl.load(X + base + offs_n, mask=mask_n, other=float("-inf"))
                x_bits = x.to(x_utype, bitcast=True)
                x_key = fpval_to_key(x_bits)
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                safe_digit = tl.where(valid, digit, RADIX_SIZE)
                hist = tl.histogram(safe_digit, RADIX_SIZE)
                counts += hist

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            cond = cumsum_desc >= k_to_find
            selected = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.max(tl.where(bins == (selected + 1), cumsum_desc, 0), axis=0)
            selected_u = selected.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos)
            k_to_find = k_to_find - counts_gt

        thr_key = desired
        thr_bits = key_to_fpval(thr_key)
        thr_val = thr_bits.to(x_dtype, bitcast=True)

        running_total = tl.zeros((), dtype=tl.int32)
        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x = tl.load(X + base + offs_n, mask=mask_n, other=float("-inf"))
            take_gt = mask_n & (x > thr_val)
            local_pos = tl.cumsum(take_gt.to(tl.int32), axis=0)
            global_pos = local_pos + running_total - 1
            write_mask = take_gt & (global_pos < k_limit) & (global_pos >= 0)
            tl.store(Yv + row * stride_ym + global_pos, x, mask=write_mask)
            tl.store(Yi + row * stride_ym + global_pos, offs_n.to(tl.int32), mask=write_mask)
            running_total += tl.sum(take_gt.to(tl.int32))

        if running_total < k_limit:
            for t in tl.range(0, n_tiles):
                if running_total < k_limit:
                    offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                    mask_n = offs_n < n_cols
                    x = tl.load(X + base + offs_n, mask=mask_n, other=float("-inf"))
                    take_eq = mask_n & (x == thr_val)
                    local_pos = tl.cumsum(take_eq.to(tl.int32), axis=0)
                    global_pos = local_pos + running_total - 1
                    write_mask = take_eq & (global_pos < k_limit) & (global_pos >= 0)
                    tl.store(Yv + row * stride_ym + global_pos, x, mask=write_mask)
                    tl.store(Yi + row * stride_ym + global_pos, offs_n.to(tl.int32), mask=write_mask)
                    running_total += tl.sum(take_eq.to(tl.int32))


# ---------------------------------------------------------------------------
# Kernel 2: TLE-optimized radix-select (SMEM L1 cache, BLOCK_N = N)
# ---------------------------------------------------------------------------

# DSM usable limit for SMEM alloc. GCU400 HW: 917KB, but the compiler's
# MRADIX/miota lowering (4-warp gather/broadcast) consumes additional DSM
# internally. Empirically BLOCK_N <= 16384 works; 32768 may overflow.
DSM_SMEM_LIMIT_ELEMS = 16384


@triton.jit
def topk_kernel_tle(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_rows,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    NUM_DIGITS: tl.constexpr,
):
    """BLOCK_N == N (no tiling). Entire row cached in SMEM."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
    x_utype = tl.dtype(f"uint{x_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS  # type: ignore[assignment, operator]
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1  # type: ignore[assignment]
    bins = tl.arange(0, RADIX_SIZE)
    offs_n = tl.arange(0, BLOCK_N)

    START_POS: tl.constexpr = (NUM_DIGITS - 1) * RADIX_BITS  # type: ignore[operator]

    smem_keys = tle.gpu.alloc(
        (BLOCK_N, ),
        dtype=tl.uint32,
        layout=None,  # type: ignore[arg-type]
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_key_ptrs = tle.gpu.local_ptr(smem_keys, (offs_n, ))
    for row in tl.range(pid, n_rows, num_pids):
        base = row * stride_xm
        mask_n = offs_n < n_cols

        x_raw = tl.load(X + base + offs_n, mask=mask_n, other=float("-inf"))
        x_bits = x_raw.to(x_utype, bitcast=True)
        x_key = fpval_to_key(x_bits)
        tl.store(smem_key_ptrs, x_key.to(tl.uint32), mask=mask_n)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        k_limit = tl.full((), K, dtype=tl.int32)

        for digit_pos in tl.static_range(START_POS, -1, -RADIX_BITS):
            cached_key = tl.load(smem_key_ptrs, mask=mask_n, other=0).to(x_utype)
            matches = (cached_key & desired_mask) == desired
            digit = ((cached_key >> digit_pos) & RADIX_MASK).to(tl.int32)
            valid = mask_n & matches
            safe_digit = tl.where(valid, digit, RADIX_SIZE)
            counts = tl.histogram(safe_digit, RADIX_SIZE)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            cond = cumsum_desc >= k_to_find
            selected = tl.max(tl.where(cond, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.max(tl.where(bins == (selected + 1), cumsum_desc, 0), axis=0)
            selected_u = selected.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos)
            k_to_find = k_to_find - counts_gt

        thr_key = desired
        cached_bits = key_to_fpval(cached_key)
        cached_val = cached_bits.to(x_dtype, bitcast=True)

        take_gt = mask_n & (cached_key > thr_key)
        local_pos = tl.cumsum(take_gt.to(tl.int32), axis=0)
        global_pos = local_pos - 1
        write_mask = take_gt & (global_pos < k_limit) & (global_pos >= 0)
        tl.store(Yv + row * stride_ym + global_pos, cached_val, mask=write_mask)
        tl.store(Yi + row * stride_ym + global_pos, offs_n.to(tl.int32), mask=write_mask)

        running_total = tl.sum(take_gt.to(tl.int32))
        take_eq = mask_n & (cached_key == thr_key)
        local_pos_eq = tl.cumsum(take_eq.to(tl.int32), axis=0)
        global_pos_eq = local_pos_eq + running_total - 1
        write_mask_eq = take_eq & (global_pos_eq < k_limit) & (global_pos_eq >= 0)
        tl.store(Yv + row * stride_ym + global_pos_eq, cached_val, mask=write_mask_eq)
        tl.store(Yi + row * stride_ym + global_pos_eq, offs_n.to(tl.int32), mask=write_mask_eq)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

GRID_SIZE = 24
NUM_WARPS = 4


def triton_naive_topk(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
):
    assert x.is_cuda or str(x.device).startswith("gcu"), "input must be on CUDA/GCU"
    assert x.ndim == 2, "input must be 2D (M, N)"
    M, N = x.shape
    if k > N:
        raise ValueError(f"k={k} must be <= N={N}")

    if out_vals is None:
        y_vals = torch.empty((M, k), device=x.device, dtype=x.dtype)
    else:
        y_vals = out_vals
    if out_idx is None:
        y_idx = torch.empty((M, k), device=x.device, dtype=torch.int32)
    else:
        y_idx = out_idx

    if N >= 8192:
        block_n = 4096
    elif N >= 2048:
        block_n = 2048
    elif N >= 512:
        block_n = 1024
    else:
        block_n = max(32, triton.next_power_of_2(N))

    x_nbits = x.dtype.itemsize * 8
    radix_bits = 5
    num_digits = (x_nbits + radix_bits - 1) // radix_bits

    topk_kernel_naive[(GRID_SIZE, )](
        x, y_vals, y_idx, x.stride(0), y_vals.stride(0), M, N, K=k, BLOCK_N=block_n,
        RADIX_BITS=radix_bits,  # type: ignore[arg-type]
        NUM_DIGITS=num_digits, num_warps=NUM_WARPS, num_stages=1,  # type: ignore[call-arg]
    )
    return y_vals, y_idx


def triton_tle_topk(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
):
    assert x.is_cuda or str(x.device).startswith("gcu"), "input must be on CUDA/GCU"
    assert x.ndim == 2, "input must be 2D (M, N)"
    M, N = x.shape
    if k > N:
        raise ValueError(f"k={k} must be <= N={N}")

    block_n = triton.next_power_of_2(N)

    if block_n > DSM_SMEM_LIMIT_ELEMS:
        raise ValueError(f"N={N} (block_n={block_n}) exceeds DSM SMEM limit "
                         f"({DSM_SMEM_LIMIT_ELEMS} elems). Need compiler optimization "
                         f"to reduce MRADIX/miota DSM overhead.")

    if out_vals is None:
        y_vals = torch.empty((M, k), device=x.device, dtype=x.dtype)
    else:
        y_vals = out_vals
    if out_idx is None:
        y_idx = torch.empty((M, k), device=x.device, dtype=torch.int32)
    else:
        y_idx = out_idx

    x_nbits = x.dtype.itemsize * 8
    radix_bits = 5
    num_digits = (x_nbits + radix_bits - 1) // radix_bits

    topk_kernel_tle[(GRID_SIZE, )](
        x, y_vals, y_idx, x.stride(0), y_vals.stride(0), M, N, K=k, BLOCK_N=block_n,
        RADIX_BITS=radix_bits,  # type: ignore[arg-type]
        NUM_DIGITS=num_digits, num_warps=NUM_WARPS, num_stages=1,  # type: ignore[call-arg]
    )
    return y_vals, y_idx


# ---------------------------------------------------------------------------
# Correctness & benchmark
# ---------------------------------------------------------------------------


def _get_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


SHAPES = [
    (64, 128, 8),
    (64, 1024, 32),
    (64, 8192, 128),
    (128, 16384, 256),
]


def run_correctness(m: int, n: int, k: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.rand((m, n), device=DEVICE, dtype=dtype)
    t_vals, _ = torch.topk(x, k, dim=1, sorted=False)
    t_vals_sorted = torch.sort(t_vals, dim=1, descending=True).values

    # Naive kernel
    y_vals, y_idx = triton_naive_topk(x, k)
    y_vals_sorted = torch.sort(y_vals, dim=1, descending=True).values
    torch.testing.assert_close(y_vals_sorted, t_vals_sorted, rtol=1e-3, atol=1e-3)
    gathered = x.gather(1, y_idx.to(torch.int64))
    torch.testing.assert_close(gathered, y_vals, rtol=1e-3, atol=1e-3)
    print(f"  PASS  naive  M={m:4d} N={n:5d} K={k:3d}")

    # TLE kernel (skip if DSM overflow)
    block_n = triton.next_power_of_2(n)
    if block_n > DSM_SMEM_LIMIT_ELEMS:
        print(f"  SKIP  tle    M={m:4d} N={n:5d} K={k:3d}  (DSM overflow: block_n={block_n})")
        return
    y_vals, y_idx = triton_tle_topk(x, k)
    y_vals_sorted = torch.sort(y_vals, dim=1, descending=True).values
    torch.testing.assert_close(y_vals_sorted, t_vals_sorted, rtol=1e-3, atol=1e-3)
    gathered = x.gather(1, y_idx.to(torch.int64))
    torch.testing.assert_close(gathered, y_vals, rtol=1e-3, atol=1e-3)
    print(f"  PASS  tle    M={m:4d} N={n:5d} K={k:3d}")


if "--only_unit_test" in sys.argv:
    _dtype = _get_dtype("float32")
    run_correctness(8, 128, 8, _dtype)
    sys.exit(0)

_BENCH_PROVIDERS = ["naive", "tle", "torch"]
_BENCH_NAMES = ["Triton-Naive-Radix", "Triton-TLE-L1Cache", "Torch-TopK"]
_BENCH_STYLES = [("red", "-"), ("blue", "-"), ("orange", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=SHAPES,
        x_log=True,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="topk-naive-vs-tle-vs-torch",
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
    if provider == "naive":

        def run_kernel():
            triton_naive_topk(x, K, out_vals=y_vals, out_idx=y_idx)

        ms, min_ms, max_ms = triton.testing.do_bench(run_kernel, quantiles=quantiles, warmup=bench_warmup,
                                                     rep=bench_rep)
    elif provider == "tle":
        block_n = triton.next_power_of_2(N)
        if block_n > DSM_SMEM_LIMIT_ELEMS:
            return float("nan"), float("nan"), float("nan")

        def run_kernel():
            triton_tle_topk(x, K, out_vals=y_vals, out_idx=y_idx)

        ms, min_ms, max_ms = triton.testing.do_bench(run_kernel, quantiles=quantiles, warmup=bench_warmup,
                                                     rep=bench_rep)
    else:

        def run_kernel():
            torch.topk(x, K, dim=1, sorted=False)

        ms, min_ms, max_ms = triton.testing.do_bench(run_kernel, quantiles=quantiles, warmup=bench_warmup,
                                                     rep=bench_rep)
    return ms, max_ms, min_ms


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--skip_correctness", action="store_true")
    parser.add_argument("--show_plots", action="store_true")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)
    if not args.skip_correctness:
        for m, n, k in [(8, 128, 8), (8, 256, 16), (16, 1024, 32)] + SHAPES:
            run_correctness(m, n, k, dtype)

    benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype)


if __name__ == "__main__":
    main()
