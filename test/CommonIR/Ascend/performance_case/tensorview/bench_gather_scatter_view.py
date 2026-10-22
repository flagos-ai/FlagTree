"""Benchmark GatherScatterView DMA paths against equal-element-count baselines.

Gather pairs compare different access patterns/layouts, not identical operators.
Scatter compares one tile, launched by one program, against masked tl.store.
Each timed side is checked against its own CPU reference before profiling.

Established floors detect ratio regressions beyond --tolerance. A None floor
means the workload changed and needs hardware recalibration; normal checking
fails rather than treating that case as passing. --update-floors only prints
suggestions for review, never edits this file or excuses correctness failures,
invalid timings, or regressions against existing floors.

    python bench_gather_scatter_view.py                  # check against floors
    python bench_gather_scatter_view.py --csv out.csv     # also write a row per case
    python bench_gather_scatter_view.py --update-floors   # suggest new floors

Historical floors were measured on Ascend910B4-1 / CANN 9.0.0. Only compare
before/after results for the same workload and environment. Historical analysis
lives in `documents/commonir/tensorview_known_issues.md`.
"""

import argparse
import csv
import math
import statistics as st
import sys

import torch
import torch_npu  # noqa: F401  (registers the npu device)
import triton
import triton.language as tl
import triton.experimental.tle as tle
from triton.backends.ascend.testing import do_bench_npu

# Retain established floors only for unchanged workloads. Recalibrate changed
# workloads in a separate, measured update rather than guessing new floors.
# Strided/scatter recalibrated 2026-10-13 on Ascend910B4-1 / CANN 9.0.0:
# floor(0.85 * minimum ratio across three runs), warmup/active/repeats=25/50/5.
FLOOR_NOTE = "established floors retained; None means recalibration required"

WARMUP = 25
ACTIVE = 50
REPEATS = 5


def require_positive(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def bench(fn, actual, expected):
    """Check once, then return median and sample SD of REPEATS profiler means."""
    fn()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=False)
    samples = []
    for repeat in range(REPEATS):
        sample = do_bench_npu(fn, warmup=WARMUP, active=ACTIVE)
        require_positive(f"timing sample {repeat}", sample)
        samples.append(sample)
    return st.median(samples), st.stdev(samples)


def parse_tolerance(text):
    value = float(text)
    if not math.isfinite(value) or not 0 <= value < 1:
        raise argparse.ArgumentTypeError("tolerance must be finite and in [0, 1)")
    return value


def row_verdict(row, tolerance):
    # Preserve the existing slow_ms/slow_sd CSV names for the baseline side.
    for key in ("fast_ms", "slow_ms", "ratio"):
        require_positive(key, row[key])
    for key in ("fast_sd", "slow_sd"):
        if not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative, got {row[key]}")
    if row["floor"] is None:
        return "UNCALIBRATED"
    require_positive("floor", row["floor"])
    return "ok" if row["ratio"] >= row["floor"] * (1 - tolerance) else "REGRESSED"


# --------------------------------------------------------------------- kernels
# Naming: the dim arrangement, S for a sparse dim, D for a dense one. The
# payload is the trailing D run; everything before it is one loop level each.


@triton.jit
def k_sd_fast(src, out, i0, D0: tl.constexpr, D1: tl.constexpr, T0: tl.constexpr):
    """S D -- sparse on dim 0, payload on dim 1. Takes the row DMA."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1], strides=[D1, 1], tile=[T0, D1], sparse_dim=[0])
    a = tl.load(i0 + tl.arange(0, T0))
    t = tle.tile_load(v, index=(a, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    rows = tl.arange(0, T0)[:, None]
    cols = tl.arange(0, D1)[None, :]
    tl.store(out + rows * D1 + cols, t2)


@triton.jit
def k_ssd_fast(src, out, i0, i1, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr,
               T1: tl.constexpr, T2: tl.constexpr):
    """S S D -- two sparse dims (a prefix), payload on dim 2. Block DMA."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, T1, T2],
                                     sparse_dim=[0, 1])
    a = tl.load(i0 + tl.arange(0, T0))
    c = tl.load(i1 + tl.arange(0, T1))
    t = tle.tile_load(v, index=(a, c, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_sds_scalar(src, out, i0, i2, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr,
                 T1: tl.constexpr, T2: tl.constexpr):
    """S D S -- sparse innermost, so there is no payload: ScalarTransfer.

    This is the comparison baseline for k_ssd_fast: same element count, same
    rank, but `payload.isScalar()` so the gate rejects the block DMA.
    """
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, T1, T2],
                                     sparse_dim=[0, 2])
    a = tl.load(i0 + tl.arange(0, T0))
    c = tl.load(i2 + tl.arange(0, T2))
    t = tle.tile_load(v, index=(a, 0, c), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_dsd_fast(src, out, i1, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr, T1: tl.constexpr,
               T2: tl.constexpr):
    """D S D -- the sparse dim is not a prefix; dim 0 is a plain loop level."""
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, D2, 1], tile=[T0, T1, T2],
                                     sparse_dim=[1])
    c = tl.load(i1 + tl.arange(0, T1))
    t = tle.tile_load(v, index=(0, c, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_strided_fast(src, out, i0, D0: tl.constexpr, D1: tl.constexpr, PITCH: tl.constexpr, T0: tl.constexpr,
                   T2: tl.constexpr):
    """Padded pitch: rows are PITCH wide but only T2 of them are the payload.

    stride[1] = PITCH != T2 * 1, so the payload is strided rather than densely
    packed -- one strided DMA, not the scalar path.
    """
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, PITCH], strides=[D1 * PITCH, PITCH, 1], tile=[T0, D1, T2],
                                     sparse_dim=[0])
    a = tl.load(i0 + tl.arange(0, T0))
    t = tle.tile_load(v, index=(a, 0, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (D1 * T2) + tl.arange(0, D1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_transposed_scalar(src, out, i0, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, T0: tl.constexpr,
                        T1: tl.constexpr, T2: tl.constexpr):
    """Transposed strides -- innermost payload stride != 1.

    Baseline for k_strided_fast: the logical block is a transpose of the memory
    layout, which no single DMA expresses, so this must stay on the scalar path.
    """
    v = tle.make_gather_scatter_view(src, shape=[D0, D1, D2], strides=[D1 * D2, 1, D1], tile=[T0, T1, T2],
                                     sparse_dim=[0])
    a = tl.load(i0 + tl.arange(0, T0))
    t = tle.tile_load(v, index=(a, 0, 0), dst_space=tle.language.dsa.ascend.UB)
    t2 = t * 1.0
    off = (tl.arange(0, T0)[:, None, None] * (T1 * T2) + tl.arange(0, T1)[None, :, None] * T2 +
           tl.arange(0, T2)[None, None, :])
    tl.store(out + off, t2)


@triton.jit
def k_scatter_gsv(src, dst, i0, D0: tl.constexpr, D1: tl.constexpr, T0: tl.constexpr):
    """Scatter via the view: no mask operand, so the store keeps its row DMA."""
    rows = tl.arange(0, T0)[:, None]
    cols = tl.arange(0, D1)[None, :]
    val = tl.load(src + rows * D1 + cols)
    a = tl.load(i0 + tl.arange(0, T0))
    v = tle.make_gather_scatter_view(dst, shape=[D0, D1], strides=[D1, 1], tile=[T0, D1], sparse_dim=[0])
    tle.tile_store(v, value=val, index=(a, 0), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def k_scatter_masked(src, dst, i0, D0, D1: tl.constexpr, T0: tl.constexpr):
    """Baseline: scattered store with a data-dependent, two-sided bounds mask."""
    rows = tl.arange(0, T0)[:, None]
    cols = tl.arange(0, D1)[None, :]
    val = tl.load(src + rows * D1 + cols)
    a = tl.load(i0 + tl.arange(0, T0)).to(tl.int64)
    ok = (a >= 0) & (a < D0)
    tl.store(dst + a[:, None] * D1 + cols, val, mask=ok[:, None])


# ----------------------------------------------------------------------- cases


def case_multi_sparse():
    """SSD vs SDS: equal element counts, different source accesses."""
    out = []
    for D0, D1, D2, T0, T1, T2, floor in [
        (16, 32, 64, 4, 8, 64, 10.0),
        (32, 64, 128, 8, 16, 128, 22.0),
        (16, 32, 256, 8, 16, 256, 35.0),
    ]:
        src = torch.randn((D0, D1, D2), dtype=torch.float32, device="npu")
        o = torch.full((T0, T1, T2), float("nan"), device="npu")
        o2 = torch.full_like(o, float("nan"))
        i0 = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
        i1 = torch.randint(0, D1, (T1, ), dtype=torch.int32, device="npu")
        i2 = torch.randint(0, D2, (T2, ), dtype=torch.int32, device="npu")
        host = src.cpu()
        a, b, c = i0.cpu().long(), i1.cpu().long(), i2.cpu().long()
        expected_fast = host.index_select(0, a).index_select(1, b)[:, :, :T2]
        expected_slow = host.index_select(0, a)[:, :T1, :].index_select(2, c)
        fast, fsd = bench(lambda: k_ssd_fast[(1, )](src, o, i0, i1, D0, D1, D2, T0, T1, T2), o, expected_fast)
        slow, ssd = bench(lambda: k_sds_scalar[(1, )](src, o2, i0, i2, D0, D1, D2, T0, T1, T2), o2, expected_slow)
        out.append(
            dict(case="multi-sparse block DMA", shape=f"tile[{T0},{T1},{T2}]", fast_ms=fast, fast_sd=fsd, slow_ms=slow,
                 slow_sd=ssd, ratio=slow / fast, floor=floor))
    return out


def case_non_prefix_sparse():
    """DSD vs SDS: equal tile sizes, different sparse dimensions/accesses."""
    D0, D1, D2, T0, T1, T2 = 16, 32, 128, 4, 8, 128
    src = torch.randn((D0, D1, D2), dtype=torch.float32, device="npu")
    o = torch.full((T0, T1, T2), float("nan"), device="npu")
    o2 = torch.full_like(o, float("nan"))
    i1 = torch.randint(0, D1, (T1, ), dtype=torch.int32, device="npu")
    i2 = torch.randint(0, D2, (T2, ), dtype=torch.int32, device="npu")
    i0 = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
    host = src.cpu()
    a, b, c = i0.cpu().long(), i1.cpu().long(), i2.cpu().long()
    expected_fast = host[:T0, :, :T2].index_select(1, b)
    expected_slow = host.index_select(0, a)[:, :T1, :].index_select(2, c)
    fast, fsd = bench(lambda: k_dsd_fast[(1, )](src, o, i1, D0, D1, D2, T0, T1, T2), o, expected_fast)
    slow, ssd = bench(lambda: k_sds_scalar[(1, )](src, o2, i0, i2, D0, D1, D2, T0, T1, T2), o2, expected_slow)
    return [
        dict(case="D S D (non-prefix sparse)", shape=f"tile[{T0},{T1},{T2}]", fast_ms=fast, fast_sd=fsd, slow_ms=slow,
             slow_sd=ssd, ratio=slow / fast, floor=19.0)
    ]


def case_strided_payload():
    """A padded pitch still rides one strided DMA; only a transposed layout
    has to fall back."""
    D0, D1, PITCH, T0, T2 = 32, 8, 72, 8, 64
    src = torch.randn((D0, D1, PITCH), dtype=torch.float32, device="npu")
    o = torch.full((T0, D1, T2), float("nan"), device="npu")
    i0 = torch.randint(0, D0, (T0, ), dtype=torch.int32, device="npu")
    a = i0.cpu().long()
    expected_fast = src.cpu().index_select(0, a)[:, :, :T2]
    fast, fsd = bench(lambda: k_strided_fast[(1, )](src, o, i0, D0, D1, PITCH, T0, T2), o, expected_fast)

    D2 = T2  # Both sides move T0 * D1 * T2 = 4096 elements.
    src2 = torch.randn((D0, D1, D2), dtype=torch.float32, device="npu")
    o2 = torch.full_like(o, float("nan"))
    # Interpret the physical allocation using the view's [D1*D2, 1, D1] strides.
    logical = src2.cpu().reshape(D0, D2, D1).transpose(1, 2)
    expected_slow = logical.index_select(0, a)
    slow, ssd = bench(lambda: k_transposed_scalar[(1, )](src2, o2, i0, D0, D1, D2, T0, D1, D2), o2, expected_slow)
    return [
        dict(case="strided payload vs transposed", shape=f"pitch={PITCH} tile[{T0},{D1},{T2}]", fast_ms=fast,
             fast_sd=fsd, slow_ms=slow, slow_sd=ssd, ratio=slow / fast, floor=12.0)
    ]


def case_scatter():
    """One scatter tile with unique, all-valid indices on both timed sides."""
    out = []
    for D0, D1, T0, floor in [
        (1024, 128, 32, 91.0),
        (4096, 256, 32, 174.0),
    ]:
        src = torch.randn((T0, D1), dtype=torch.float32, device="npu")
        i0 = torch.randperm(D0, device="npu")[:T0].to(torch.int32)
        # Check the full destination, including untouched rows and both canaries.
        guard = T0 * D1
        initial = torch.full((guard + D0 * D1 + guard, ), -7.0)
        initial[guard:-guard] = 0.0
        expected = initial.clone()
        expected[guard:-guard].view(D0, D1)[i0.cpu().long()] = src.cpu()
        backing1, backing2 = initial.to("npu"), initial.to("npu")
        d1 = backing1[guard:-guard].view(D0, D1)
        d2 = backing2[guard:-guard].view(D0, D1)
        grid = (1, )  # Neither kernel uses program_id: exactly one tile per launch.
        fast, fsd = bench(lambda: k_scatter_gsv[grid](src, d1, i0, D0, D1, T0), backing1, expected)
        slow, ssd = bench(lambda: k_scatter_masked[grid](src, d2, i0, D0, D1, T0), backing2, expected)
        out.append(
            dict(case="scatter: view vs masked store", shape=f"[{D0},{D1}] tile[{T0},{D1}]", fast_ms=fast, fast_sd=fsd,
                 slow_ms=slow, slow_sd=ssd, ratio=slow / fast, floor=floor))
    return out


CASES = [
    ("multi-sparse", case_multi_sparse),
    ("non-prefix sparse", case_non_prefix_sparse),
    ("strided payload", case_strided_payload),
    ("scatter", case_scatter),
]


def main():
    ap = argparse.ArgumentParser(description="GatherScatterView DMA paths vs equal-work baselines")
    ap.add_argument("--csv", help="write one row per case")
    ap.add_argument("--tolerance", type=parse_tolerance, default=0.15,
                    help="allowed fractional drop below a floor (default 0.15)")
    ap.add_argument("--update-floors", action="store_true",
                    help="print suggested floors; allow uncalibrated cases, but not failures")
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 96)
    print("GatherScatterView: block DMA vs baseline    "
          f"(warmup={WARMUP} active={ACTIVE} repeats={REPEATS}, median of means)")
    print("latencies in ms; +/- is sample SD across the profiler means")
    print(f"floors: {FLOOR_NOTE}; failing below floor*(1-{args.tolerance})")
    print("=" * 96)

    rows = []
    for name, fn in CASES:
        print(f"\n[{name}]")
        try:
            results = fn()
            for r in results:
                r["status"] = row_verdict(r, args.tolerance)
        except (AssertionError, ValueError, RuntimeError) as error:
            print(f"FAILED [{name}]: {error}", file=sys.stderr)
            return 1
        for r in results:
            rows.append(r)
            floor = "pending" if r["floor"] is None else f"{r['floor']:.1f}x"
            print(f"  {r['shape']:<30} fast={r['fast_ms']:.6f}±{r['fast_sd']:.6f}"
                  f"  baseline={r['slow_ms']:.6f}±{r['slow_sd']:.6f}"
                  f"  {r['ratio']:>7.1f}x  floor={floor}  {r['status']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.update_floors:
        print("\nsuggested floors (~15% under this run; review before applying):")
        for r in rows:
            print(f"  {r['case']:<32} {r['shape']:<30} {r['ratio'] * 0.85:.6g}")

    bad = [r for r in rows if r["status"] == "REGRESSED"]
    pending = [r for r in rows if r["status"] == "UNCALIBRATED"]
    print("\n" + "=" * 96)
    print(f"{len(rows) - len(bad) - len(pending)}/{len(rows)} cases at or above floor; "
          f"{len(pending)} require calibration")
    for r in bad:
        print(f"  REGRESSED {r['case']} {r['shape']}: "
              f"{r['ratio']:.1f}x < floor {r['floor']:.0f}x")
    if pending:
        print("Uncalibrated cases are not regression passes; measure and review new floors.")
    print("=" * 96)
    return 1 if bad or (pending and not args.update_floors) else 0


if __name__ == "__main__":
    sys.exit(main())
