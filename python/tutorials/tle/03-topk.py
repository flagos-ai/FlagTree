"""
Top-K with Triton (Radix-Only Tutorial)
=======================================

This tutorial implements Top-K over the last dimension of an (M, N) tensor and
compares:
- radix: Triton radix-select kernel
- torch: torch.topk
"""

import argparse
import sys

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.gpu as tle

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
def topk_kernel_radix_triton(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_cols,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RADIX_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    # Stage 0: setup dtype metadata and key packing types.
    x_dtype = X.dtype.element_ty
    x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_utype = tl.dtype(f"uint{x_nbits}")
    x_ultype = tl.dtype(f"uint{y_nbits}")

    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
    bins = tl.arange(0, RADIX_SIZE)
    one = tl.full([BLOCK_N], 1, tl.int32)

    desired = tl.full((), 0, dtype=x_utype)
    desired_mask = tl.full((), 0, dtype=x_utype)
    k_to_find = tl.full((), K, dtype=tl.int32)
    n_tiles = tl.cdiv(n_cols, BLOCK_N)

    # Stage 1: shared-memory histogram storage for each radix digit.
    smem_counts = tle.alloc(
        [RADIX_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    smem_count_ptrs = tle.local_ptr(smem_counts, (bins, ))

    # Stage 2: MSD radix-select; pick one digit bucket per pass.
    for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
        tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = fpval_to_key_with_nan(x, x_bits)
            matches = (x_key & desired_mask) == desired
            digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
            valid = mask_n & matches
            count_addrs = tle.local_ptr(smem_counts, (digit, ))
            tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

        counts = tl.load(smem_count_ptrs)

        # Compute descending cumulative histogram in-place.
        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
        tl.store(smem_count_ptrs, cumsum_desc)

        selected_scalar = 0
        counts_gt_scalar = 0
        found = 0
        for rev in tl.static_range(RADIX_SIZE):
            d = RADIX_SIZE - 1 - rev
            cum_d = tl.load(tle.local_ptr(smem_counts, (d, )))
            if d + 1 < RADIX_SIZE:
                cum_next = tl.load(tle.local_ptr(smem_counts, (d + 1, )))
            else:
                cum_next = 0
            take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
            selected_scalar = tl.where(take, d, selected_scalar)
            counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
            found = tl.where(take, 1, found)

        selected_u = selected_scalar.to(x_utype)
        desired = desired | (selected_u << digit_pos)
        desired_mask = desired_mask | (tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos)
        k_to_find = k_to_find - counts_gt_scalar

    # Stage 3: compact candidates with shared-memory atomic write count.
    thr_key = desired

    min_val = tl.full((), float("-inf"), tl.float32).to(x_dtype)
    min_bits = min_val.to(x_utype, bitcast=True)
    min_key = fpval_to_key_with_nan(min_val, min_bits)
    min_packed = min_key.to(x_ultype) << 16
    offs_k = tl.arange(0, K_PAD)

    smem_selected = tle.alloc(
        [K_PAD],
        dtype=x_ultype,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    smem_selected_ptrs = tle.local_ptr(smem_selected, (offs_k, ))
    tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

    smem_write_count = tle.alloc(
        [1],
        dtype=tl.int32,
        layout=None,
        scope=tle.smem,
        nv_mma_shared_layout=False,
    )
    tl.store(tle.local_ptr(smem_write_count, (0, )), 0)
    write_count_ptrs = tle.local_ptr(smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32), ))

    # Pass 1: write all values strictly greater than threshold.
    for t in tl.range(0, n_tiles):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        x_ptrs = X + pid * stride_xm + offs_n
        x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
        x_bits = x.to(x_utype, bitcast=True)
        x_key = fpval_to_key_with_nan(x, x_bits)
        idx_key = (n_cols - offs_n).to(x_ultype)
        packed = (x_key.to(x_ultype) << 16) | idx_key
        take_gt = mask_n & (x_key > thr_key)
        pos = tl.atomic_add(write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta")
        write_mask = take_gt & (pos < K_PAD)
        dst_ptrs = tle.local_ptr(smem_selected, (pos.to(tl.int32), ))
        tl.store(dst_ptrs, packed, mask=write_mask)

    # Pass 2: fill remaining slots with values equal to threshold (first-come-first-serve).
    for t in tl.range(0, n_tiles):
        offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_cols
        x_ptrs = X + pid * stride_xm + offs_n
        x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
        x_bits = x.to(x_utype, bitcast=True)
        x_key = fpval_to_key_with_nan(x, x_bits)
        idx_key = (n_cols - offs_n).to(x_ultype)
        packed = (x_key.to(x_ultype) << 16) | idx_key
        take_eq = mask_n & (x_key == thr_key)
        pos = tl.atomic_add(write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta")
        write_mask = take_eq & (pos < K_PAD)
        dst_ptrs = tle.local_ptr(smem_selected, (pos.to(tl.int32), ))
        tl.store(dst_ptrs, packed, mask=write_mask)

    selected_packed = tl.load(smem_selected_ptrs)

    # Stage 4: unpack final packed keys and write outputs.
    topk = tl.sort(selected_packed, dim=0, descending=True)
    idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
    idx_raw = (topk & idx_mask).to(tl.uint32)
    y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int32)
    y_values_raw = (topk >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    mask_k = offs_k < K
    yv_ptrs = Yv + pid * stride_ym + offs_k
    yi_ptrs = Yi + pid * stride_ym + offs_k
    tl.store(yv_ptrs, y_values, mask=mask_k)
    tl.store(yi_ptrs, y_indices, mask=mask_k)


def triton_radix_topk(
    x: torch.Tensor,
    k: int,
    out_vals: torch.Tensor | None = None,
    out_idx: torch.Tensor | None = None,
):
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")
    if n_cols > 65535:
        raise ValueError(f"N={n_cols} too large for 16-bit packed index encoding (max 65535)")

    k_pad = triton.next_power_of_2(k)
    if k_pad > 1024:
        raise ValueError(f"k={k} too large for radix kernel (K_PAD={k_pad}, max 1024)")

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
    block_n_radix = max(k_pad, min(512, triton.next_power_of_2(n_cols)))
    block_n_radix = min(block_n_radix, 1024)
    topk_kernel_radix_triton[(num_blocks, )](
        x,
        y_vals,
        y_idx,
        x.stride(0),
        y_vals.stride(0),
        n_cols,
        K=k,
        K_PAD=k_pad,
        BLOCK_N=block_n_radix,
        RADIX_BITS=4,
        num_warps=4,
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
    t_vals, _ = torch.topk(x, k, dim=1)
    y_vals, y_idx = triton_radix_topk(x, k)
    torch.testing.assert_close(y_vals, t_vals, rtol=1e-3, atol=1e-3)
    gathered = x.gather(1, y_idx.to(torch.int64))
    torch.testing.assert_close(gathered, y_vals, rtol=1e-3, atol=1e-3)
    print("Correctness check passed (radix).")


if "--only_unit_test" in sys.argv:
    _args = argparse.Namespace(batch=8, seq_len=128, K=2, dtype="float16")
    _dtype = _get_dtype(_args.dtype)
    run_correctness(_args.batch, _args.seq_len, _args.K, _dtype)
    sys.exit(0)

_BENCH_PROVIDERS = ["radix", "torch"]
_BENCH_NAMES = ["Triton-RadixSelect", "Torch-TopK"]
_BENCH_STYLES = [("red", "-"), ("orange", "-")]


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
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-topk-radix-vs-torch",
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
        if N > 65535:
            return float("nan"), float("nan"), float("nan")
        k_pad = triton.next_power_of_2(K)
        if k_pad > 1024:
            return float("nan"), float("nan"), float("nan")

        def run_kernel():
            triton_radix_topk(x, K, out_vals=y_vals, out_idx=y_idx)

        ms, min_ms, max_ms = triton.testing.do_bench(
            run_kernel,
            quantiles=quantiles,
            warmup=bench_warmup,
            rep=bench_rep,
        )
    else:

        def run_kernel():
            torch.topk(x, K, dim=1)

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
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)
    check_m = args.batch
    check_n = min(args.seq_len, 256)
    check_k = min(args.K, check_n)
    run_correctness(check_m, check_n, check_k, dtype)

    benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype)


if __name__ == "__main__":
    main()
