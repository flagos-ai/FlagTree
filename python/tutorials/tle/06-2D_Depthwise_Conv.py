"""
2D Depthwise Conv with TLE extract_tile
========================================

Compares a Triton baseline depthwise conv kernel against a TLE optimized
version that uses ``tle.extract_tile``.

Layout is HWC (channels-last). Kernel size K is odd, padding = 0 (valid conv).

Usage
-----
::

    # correctness only (default)
    python python/tutorials/tle/06-2D_Depthwise_Conv.py

    # correctness + benchmark table
    python python/tutorials/tle/06-2D_Depthwise_Conv.py --benchmark

    # specify dtype
    python python/tutorials/tle/06-2D_Depthwise_Conv.py --benchmark --dtype float16
"""

import argparse
import math
import sys

import numpy as np
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def _print_env():
    """Print test environment information for reproducibility."""
    print(f"GPU: {torch.cuda.get_device_name()} | CUDA: {torch.version.cuda} | Triton: {triton.__version__}")
    print()


# Benchmark parameters
BENCH_WARMUP = 25
BENCH_REP = 100
BENCH_RUNS = 5  # number of runs for stddev


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2**math.ceil(math.log2(x))


# %%
# Kernels (Triton baseline)
# -------------------------


@triton.jit
def conv2d_baseline(
    inp_ptr,
    wgt_ptr,
    out_ptr,
    H,
    W,
    C,
    OH,
    OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    TILE_OH: tl.constexpr,
    TILE_OW: tl.constexpr,
    TILE_C: tl.constexpr,
):
    pid_oh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_c = tl.program_id(2)
    oh0 = pid_oh * TILE_OH
    ow0 = pid_ow * TILE_OW
    c0 = pid_c * TILE_C

    offs_c = c0 + tl.arange(0, TILE_C)
    offs_oh = oh0 + tl.arange(0, TILE_OH)
    offs_ow = ow0 + tl.arange(0, TILE_OW)
    c_mask = offs_c < C

    acc = tl.zeros((TILE_OH, TILE_OW, TILE_C), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            w_ptrs = wgt_ptr + (kh * KW + kw) * C + offs_c
            w = tl.load(w_ptrs, mask=c_mask, other=0.0)
            ih = offs_oh + kh
            iw = offs_ow + kw
            ih_ok = (ih >= 0) & (ih < H)
            iw_ok = (iw >= 0) & (iw < W)
            inp_ptrs = (inp_ptr + ih[:, None, None] * (W * C) + iw[None, :, None] * C + offs_c[None, None, :])
            mask = ih_ok[:, None, None] & iw_ok[None, :, None] & c_mask[None, None, :]
            x = tl.load(inp_ptrs, mask=mask, other=0.0)
            acc += x * w[None, None, :]

    out_ptrs = (out_ptr + offs_oh[:, None, None] * (OW * C) + offs_ow[None, :, None] * C + offs_c[None, None, :])
    out_mask = ((offs_oh[:, None, None] < OH) & (offs_ow[None, :, None] < OW) & c_mask[None, None, :])
    tl.store(out_ptrs, acc, mask=out_mask)


# %%
# Kernels (TLE static extract_tile)
# ---------------------------------


@triton.jit
def conv2d_extract_static(
    inp_ptr,
    wgt_ptr,
    out_ptr,
    H,
    W,
    C,
    OH,
    OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    TILE_OH: tl.constexpr,
    TILE_OW: tl.constexpr,
    TILE_C: tl.constexpr,
    HALO_H_P2: tl.constexpr,
    HALO_W_P2: tl.constexpr,
    HALO_H: tl.constexpr,
    HALO_W: tl.constexpr,
):
    pid_oh = tl.program_id(0)
    pid_ow = tl.program_id(1)
    pid_c = tl.program_id(2)
    oh0 = pid_oh * TILE_OH
    ow0 = pid_ow * TILE_OW
    c0 = pid_c * TILE_C

    offs_c = c0 + tl.arange(0, TILE_C)
    c_mask = offs_c < C
    offs_ih_p2 = oh0 + tl.arange(0, HALO_H_P2)
    offs_iw_p2 = ow0 + tl.arange(0, HALO_W_P2)

    ih_ok = (offs_ih_p2 < oh0 + HALO_H) & (offs_ih_p2 >= 0) & (offs_ih_p2 < H)
    iw_ok = (offs_iw_p2 < ow0 + HALO_W) & (offs_iw_p2 >= 0) & (offs_iw_p2 < W)

    halo_ptrs = (inp_ptr + offs_ih_p2[:, None, None] * (W * C) + offs_iw_p2[None, :, None] * C + offs_c[None, None, :])
    halo_mask = ih_ok[:, None, None] & iw_ok[None, :, None] & c_mask[None, None, :]
    halo = tl.load(halo_ptrs, mask=halo_mask, other=0.0)

    acc = tl.zeros((TILE_OH, TILE_OW, TILE_C), dtype=tl.float32)
    offs_oh = oh0 + tl.arange(0, TILE_OH)
    offs_ow = ow0 + tl.arange(0, TILE_OW)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            w_ptrs = wgt_ptr + (kh * KW + kw) * C + offs_c
            w = tl.load(w_ptrs, mask=c_mask, other=0.0)
            patch = tle.extract_tile(
                halo,
                index=[kh, kw, 0],
                tile_shape=[TILE_OH, TILE_OW, TILE_C],
                strides=[1, 1, 1],
            )
            acc += patch * w[None, None, :]

    out_ptrs = (out_ptr + offs_oh[:, None, None] * (OW * C) + offs_ow[None, :, None] * C + offs_c[None, None, :])
    out_mask = ((offs_oh[:, None, None] < OH) & (offs_ow[None, :, None] < OW) & c_mask[None, None, :])
    tl.store(out_ptrs, acc, mask=out_mask)


# %%
# Python wrappers
# ---------------


def _halo_dims(KH, KW, TOH, TOW):
    hh = TOH + KH - 1
    hw = TOW + KW - 1
    return hh, hw, _next_pow2(hh), _next_pow2(hw)


def triton_dwconv(inp, wgt, KH, KW, TOH, TOW, TC):
    H, W, C = inp.shape
    OH, OW = H - KH + 1, W - KW + 1
    out = torch.empty((OH, OW, C), device=inp.device, dtype=inp.dtype)
    grid = (math.ceil(OH / TOH), math.ceil(OW / TOW), math.ceil(C / TC))
    conv2d_baseline[grid](
        inp,
        wgt,
        out,
        H,
        W,
        C,
        OH,
        OW,
        KH=KH,
        KW=KW,
        TILE_OH=TOH,
        TILE_OW=TOW,
        TILE_C=TC,
    )
    return out


def tle_dwconv(inp, wgt, KH, KW, TOH, TOW, TC):
    H, W, C = inp.shape
    OH, OW = H - KH + 1, W - KW + 1
    out = torch.empty((OH, OW, C), device=inp.device, dtype=inp.dtype)
    hh, hw, hhp2, hwp2 = _halo_dims(KH, KW, TOH, TOW)
    grid = (math.ceil(OH / TOH), math.ceil(OW / TOW), math.ceil(C / TC))
    conv2d_extract_static[grid](
        inp,
        wgt,
        out,
        H,
        W,
        C,
        OH,
        OW,
        KH=KH,
        KW=KW,
        TILE_OH=TOH,
        TILE_OW=TOW,
        TILE_C=TC,
        HALO_H_P2=hhp2,
        HALO_W_P2=hwp2,
        HALO_H=hh,
        HALO_W=hw,
    )
    return out


def torch_dwconv(inp, wgt, KH, KW):
    C = inp.shape[2]
    x = inp.permute(2, 0, 1).unsqueeze(0).float()
    w = wgt.reshape(KH * KW, C).T.reshape(C, 1, KH, KW).float()
    y = torch.nn.functional.conv2d(x, w, groups=C, padding=0)
    return y.squeeze(0).permute(1, 2, 0).to(inp.dtype)


# %%
# Correctness check
# -----------------


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _make_input(H, W, C, KH, KW, dtype):
    torch.manual_seed(0)
    inp = torch.randn(H, W, C, device=DEVICE, dtype=dtype)
    wgt = torch.randn(KH, KW, C, device=DEVICE, dtype=dtype)
    return inp, wgt


def run_correctness(H, W, C, KH, KW, TOH, TOW, TC, dtype):
    inp, wgt = _make_input(H, W, C, KH, KW, dtype)
    ref = torch_dwconv(inp, wgt, KH, KW)
    y_triton = triton_dwconv(inp, wgt, KH, KW, TOH, TOW, TC)
    y_tle = tle_dwconv(inp, wgt, KH, KW, TOH, TOW, TC)
    atol = 1e-2 if dtype == torch.float16 else 1e-3
    torch.testing.assert_close(y_triton.float(), ref.float(), rtol=atol, atol=atol)
    torch.testing.assert_close(y_tle.float(), ref.float(), rtol=atol, atol=atol)
    print(f"  pass  H={H} W={W} C={C} K={KH}x{KW}")


if "--only_unit_test" in sys.argv:
    _args = argparse.Namespace(H=128, W=128, C=64, KH=5, KW=5, TOH=4, TOW=4, TC=64, dtype="float32")
    _dtype = _get_dtype(_args.dtype)
    run_correctness(_args.H, _args.W, _args.C, _args.KH, _args.KW, _args.TOH, _args.TOW, _args.TC, _dtype)
    sys.exit(0)

# %%
# Benchmark
# ---------

# Mirrors the FFT tutorial's "vary one dim, sweep providers" shape:
# x-axis is spatial size H=W, y-axis is ms for {triton, tle, torch}.
_BENCH_PROVIDERS = ["triton", "tle", "torch"]
_BENCH_NAMES = ["Triton", "TLE", "Torch"]
_BENCH_STYLES = [("blue", "-"), ("orange", "-"), ("green", "-")]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["H"],
        x_vals=[112, 128, 256, 512],
        x_log=False,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-dwconv-performance",
        args={"KH": 5, "KW": 5, "TOH": 4, "TOW": 4, "TC": 64, "C": 64},
    ))
def benchmark(H, C, KH, KW, TOH, TOW, TC, provider, dtype):
    inp, wgt = _make_input(H, H, C, KH, KW, dtype)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_dwconv(inp, wgt, KH, KW), quantiles=quantiles)
    elif provider == "tle":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tle_dwconv(inp, wgt, KH, KW, TOH, TOW, TC),
                                                     quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_dwconv(inp, wgt, KH, KW, TOH, TOW, TC),
                                                     quantiles=quantiles)

    return ms, max_ms, min_ms


# %%
# Main
# ----


def _bench_one(fn, warmup=BENCH_WARMUP, rep=BENCH_REP, runs=BENCH_RUNS):
    """Run do_bench multiple times with quantiles, return stats dict."""
    p50s, p20s, p80s = [], [], []
    for _ in range(runs):
        ms, p80, p20 = triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8])
        p50s.append(ms if isinstance(ms, float) else ms[0])
        p20s.append(p20 if isinstance(p20, float) else p20[0])
        p80s.append(p80 if isinstance(p80, float) else p80[0])
    p50s = np.array(p50s)
    return {
        "mean": float(p50s.mean()),
        "std": float(p50s.std()),
        "p50": float(np.median(p50s)),
        "p90": float(np.percentile(p50s, 90)),
        "min": float(np.min(p20s)),
        "max": float(np.max(p80s)),
    }


def run_benchmark_table(dtype=torch.float32):
    """Print a detailed benchmark table with latency, p50, p90, throughput, speedup, and regression."""
    H_vals = [112, 128, 256, 512]
    C, KH, KW, TOH, TOW, TC = 64, 5, 5, 4, 4, 64
    dtype_str = "fp16" if dtype == torch.float16 else "fp32"

    print(
        f"2D Depthwise Conv | H=[112,128,256,512] W=H C={C} K={KH}x{KW} | layout=HWC {dtype_str} | Warmup={BENCH_WARMUP} Rep={BENCH_REP} Runs={BENCH_RUNS}"
    )
    print()
    print(
        f"{'H':<8} {'Triton mean':<12} {'p50':<12} {'p90':<12} {'TLE mean':<12} {'p50':<12} {'p90':<12} {'Speedup':<10}"
    )

    for H in H_vals:
        inp, wgt = _make_input(H, H, C, KH, KW, dtype)

        s_b = _bench_one(lambda: triton_dwconv(inp, wgt, KH, KW, TOH, TOW, TC))
        s_t = _bench_one(lambda: tle_dwconv(inp, wgt, KH, KW, TOH, TOW, TC))

        sp = s_b["p50"] / s_t["p50"] if s_t["p50"] > 0 else 0.0

        print(f"{H:<8} {s_b['mean']:<12.4f} {s_b['p50']:<12.4f} {s_b['p90']:<12.4f} "
              f"{s_t['mean']:<12.4f} {s_t['p50']:<12.4f} {s_t['p90']:<12.4f} {sp:.2f}x")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=128, help="spatial H for correctness")
    parser.add_argument("--W", type=int, default=128, help="spatial W for correctness")
    parser.add_argument("--C", type=int, default=64, help="channels")
    parser.add_argument("--KH", type=int, default=5)
    parser.add_argument("--KW", type=int, default=5)
    parser.add_argument("--TOH", type=int, default=4)
    parser.add_argument("--TOW", type=int, default=4)
    parser.add_argument("--TC", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "fp16", "fp32"])
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    parser.add_argument("--benchmark", action="store_true", help="print detailed benchmark table")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)

    _print_env()

    print("--- correctness ---")
    for H in [112, 128, 256, 512]:
        run_correctness(H, H, args.C, args.KH, args.KW, args.TOH, args.TOW, args.TC, dtype)
    print()

    if args.benchmark:
        run_benchmark_table(dtype)
    else:
        benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype)


if __name__ == "__main__":
    main()
