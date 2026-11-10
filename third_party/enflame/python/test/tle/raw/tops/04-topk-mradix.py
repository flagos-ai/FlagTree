"""
TLE-Raw TOPS Backend: TopK with HW MRADIX Acceleration
=======================================================

One hw thread per row: the geometry of topsop's LEVEL=1 TopK (24 hw threads,
one per SIP, 4 radix subthreads each). The row is cached in the L1 buffer this
launcher allocates, so N is capped at L1_MAX_COLS = 65536 fp32; topsop's
LEVEL=1 can use the SIP's whole 896KB VDMEM.

Design (mode A of the Struct/Raw boundary: the DTE has exactly one owner, the
raw region, and it also owns the row loop):
  - Persistent kernel: grid = min(SIP_COUNT, M) blocks = 24 hw threads (one
    per SIP), no inter-thread synchronization. The row mapping
    (row_base/row_stride) is computed here with native Triton ops and passed
    into the raw region; the loop over a thread's rows lives in the raw region
    because that is what keeps the next row's DTE issued under the current
    row's compute. One call per row is expressible -- the deferred dsl_region
    lowers to a plain llvm.call, and 04-topk-mradix-struct-raw-persistent.py
    does exactly that -- it just cannot overlap the copy with the compute.
  - DTE async for the row load into L1 (one row prefetched ahead).
  - All MAX_SHIFT_NUM (7) digit passes use HW MRADIX via per-shift template
    dispatch.
  - Two-path reclaim: the common case (n_ge == K, unique floats) runs a single
    ge count + ge scatter; ties at the k-th value (n_ge > K) recount with '>'
    and split-scatter gt then eq, truncated at K.
  - Vectorized scatter: popc_mb + miota + scatter_cond.
  - 16x unrolling in the histogram and the reclaim count/scatter.
  - Out of scope: several SIPs cooperating on one row (topsop's LEVEL>=2),
    KTOP1 (k <= 10 comparison-tree kernels) and streaming RADIX; see the limits
    list in 04-topk-mradix.tops and topk_tuner() below.

Launch: grid = min(SIP_COUNT, M), no cluster dims (one hw thread per SIP, the
NORMAL-topk geometry); num_warps=4 = the 4 radix subthreads. Because rows are
walked serially inside the kernel, the wall clock is ceil(M/SIP_COUNT) * T_row,
not M/SIP_COUNT * T_row; the measured model is in the header of
04-topk-mradix.tops.

Entry points (python 04-topk-mradix.py):
  (default)  correctness suite only (what the tops-script regression runs)
  --bench    correctness + per-shape do_bench lines (torch / tle-raw, ratio,
              speedup)
  --perf     pytest perf_report benchmarks only (triton.testing.do_bench)
"""

from pathlib import Path
import os
import sys

import pytest
import torch
import triton
import triton.language as tl
from triton.experimental.tle.raw import dialect
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw

DEVICE = triton.runtime.driver.active.get_active_torch_device()
_PROPS      = torch.gcu.get_device_properties(DEVICE.index)
SIP_COUNT   = _PROPS.sip_count
NUM_BLOCKS  = _PROPS.multi_processor_count
NUM_THREADS = _PROPS.max_threads_per_multi_processor
MAX_GRID = SIP_COUNT
NUM_WARPS = 4  # One hw thread with 4 radix subthreads; 8 (hetero) unsupported.

GCU_VDMEM_SIZE = 0xE0000                  # 896KB per SIP
TCLE_SINGLE_VEC_BYTES = 512               # OACC = 512B
DTE_SLICE_MAX_INPUT_DIM_SIZE = 0x1000000  # 2^24
BATCH_RADIX_NUM_UNROLLING = 16
MAX_THREAD_NUM = 2
KTOP1_SHORTCUT_SUBTHREAD_NUM = 8
_TOPK_BPE = 4  # the edsl kernels are fp32-only.

# Radix-select capacity. The SIP's whole VDMEM is 896KB (229376 fp32 elems);
# this kernel caches the row in the double-buffered 512KB window below and keeps
# a 16KB reclaim workspace next to it, i.e. 528KB of 896KB. 65536 is the
# contract: rows above it are rejected here instead of silently overflowing
# VDMEM.
L1_MAX_COLS = 65536
# Widest k the reclaim staging supports (WS_STAGE_K_MAX in the edsl).
# The ws buffer is L1_WS_ELEMS = 4096 uint32 elements:
#   [0, 2560)            counts / prefixes / compression table (5 * fc_al_max)
#   [2560, 2560 + 2*K)   stage_v[K], stage_i[K]
#   [4093, 4096)         subthread publishes during reclaim: desired / n_ge,
#                        comp_cnt, n_gt
# so staging must stop below them: 2*K <= 4096 - 2560 - 3 = 1533 -> K <= 766,
# rounded down to a multiple of 128 words because stage_i must start
# 512B-aligned for the S5 writeback -> 640. A larger k needs a bigger ws
# (L1_WS_ELEMS here, WS_SIZE in the edsl); WS_STAGE_K_MAX / TOPK_K_MAX then
# follow (WS_SIZE - 2560 - 3) // 2. Moving the staging to HBM does not help:
# the K slots are consumed in place by the per-row writeback, so a larger k also
# pays more on that transport.
TOPK_K_MAX = 640
# The two VDMEM windows this launcher hands to the raw region, in elements of
# the allocation dtype (tl.uint32), not bytes; both must be power-of-two shapes
# for tle.gpu.alloc and together they are 528KB of the SIP's 896KB. The row
# buffer is double buffered (2 * 256KB): each half holds one row and then its
# 32-bit sort keys, converted in place -- which is why the size does not depend
# on the input dtype.
#   L1_BUF_ELEMS  == 2 * MAX_L1_CACHE   (04-topk-mradix.tops)
#   L1_WS_ELEMS   == WS_SIZE            (same file; pins the two staging
#                                        offsets and K_MAX, keep in sync)
L1_BUF_ELEMS = tl.constexpr(2 * L1_MAX_COLS)
L1_WS_ELEMS = tl.constexpr(4096)


class TopkPattern:
    """TLE-RAW mirror of topsop's TopkPattern enum.

    Same member names and values as the topsop enum. Implementation status:

      - KTOP1 family (KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT / KTOP1_BATCH_SPLIT,
        the k <= 10 lanes): not implemented in this port; dispatch routes
        their shapes to the radix kernel (see tle_raw_topk_mradix).
      - RADIX_BATCH_SPLIT: implemented as one SIP per row, which is
        topsop's LEVEL=1. topsop refines this enum further inside one kernel
        (per_row_radix_level / batch_radix_is_support); this port has a single
        radix kernel, so no refinement members exist - the label is still
        produced for LEVEL 2/3 shapes, and the dispatch decides what runs.
      - everything else: recognized so unsupported shapes fail with a
        pattern name instead of silently taking a wrong kernel; dispatch
        raises on them.
    """

    RADIX = 0               # streaming radix; unsupported
    HEAP = 1                # unsupported
    QUICK = 2               # unsupported
    BITONIC = 3             # unsupported
    KTOP1_AXIS_SPLIT = 4    # k <= 10; routed to the radix kernel
    SCAN = 5                # unsupported
    SORT = 6                # unsupported
    KTOP1_BATCH_SPLIT = 7   # k <= 10; routed to the radix kernel
    RADIX_BATCH_SPLIT = 8   # supported: one SIP per row (topsop LEVEL=1)
    KTOP1_SHORTCUT = 9      # k <= 10; routed to the radix kernel


_PATTERN_NAMES = {
    getattr(TopkPattern, name): name
    for name in dir(TopkPattern) if not name.startswith("_")
}

_KTOP1_PATTERNS = frozenset((TopkPattern.KTOP1_SHORTCUT,
                             TopkPattern.KTOP1_AXIS_SPLIT,
                             TopkPattern.KTOP1_BATCH_SPLIT))


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


@dialect(name="tops", file=Path(__file__).parent / "04-topk-mradix.tops",
         extern_func_name="topk_mradix_l1_kernel", deferred=True)
def edsl_topk_mradix_l1(*args, **kwargs):
    ...


# One @triton.jit launcher per edsl entry. A runtime dispatch if/else would be
# legal now (deferred raw calls lower in place and may sit inside scf.if), but
# this kernel has a single entry point, so it stays branch-free.
@triton.jit(do_not_specialize=["M", "N", "K"])
def topk_mradix_l1_launch(
    input_ptr,
    output_vals_ptr,
    output_idx_ptr,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    # No scratch argument: the histogram lives in MRADIX's own registers and
    # the reclaim workspace in the l1 VDMEM buffers allocated below.
    #
    # Buffer ownership: both windows are allocated HERE with tle.gpu.alloc and
    # handed to the raw region as pointers (tle.gpu.local_ptr in the call
    # below). The raw region must not also declare them __local__: both
    # allocators start at VDMEM offset 0 with no coordination, so raw static
    # buffers would silently overlap this window (measured: every shape fails in
    # the overlapping layout, passes once shifted by one raw buffer). Handing
    # the pointers over costs ~0.37us per launch, which the persistent loop
    # amortizes over this hw thread's rows.
    #
    # Row mapping (native Triton): program_id/num_programs decide which rows
    # this hw thread walks; the loop itself stays in the raw region so that the
    # next row's DTE is issued under the current row's compute.
    row_base = tl.program_id(0) * BLOCK_SIZE
    row_stride = tl.num_programs(0) * BLOCK_SIZE
    l1 = tle.gpu.alloc([L1_BUF_ELEMS], dtype=tl.uint32, scope=tle.gpu.smem)
    ws = tle.gpu.alloc([L1_WS_ELEMS],  dtype=tl.uint32, scope=tle.gpu.smem)
    tle_raw.call(
        edsl_topk_mradix_l1,
        [input_ptr, output_vals_ptr, output_idx_ptr,
         tle.gpu.local_ptr(l1, (0,)), tle.gpu.local_ptr(ws, (0,)),
         M, N, K, row_base, row_stride],
        output_indices=[1, 2])


def _batch_radix_is_support(n_cols: int, level: int) -> bool:
    """Does one row fit the VDMEM budget of `level` cooperating SIPs?

    Mirrors topsop's capacity test for the per-row radix path, fp32 only. The
    launch geometry is the same full grid for all three levels
    (<<<NUM_BLOCKS, NUM_THREADS>>>, i.e. <<<4, 6>>> = 24 SIPs on gcu400); the
    level only decides how many of those SIPs work on one row, and therefore
    how the single-SIP budget of GCU_VDMEM_SIZE/bpe scales:
      level 1: 1 SIP per row -> 24 rows in flight, 1x budget.
      level 2: one block's 6 SIPs per row -> 4 rows in flight, 6x budget.
      level 3: all 24 SIPs on one row -> 1 row in flight, 24x budget.
    At levels 2/3 the row is sliced across the SIPs: each keeps only its slice
    in VDMEM and the overflow (histogram, reclaimed candidates) lives in
    global workspace, which is why the budget can be multiplied.
    """
    vec_len = TCLE_SINGLE_VEC_BYTES // _TOPK_BPE
    tiling_num = _align_up(n_cols, BATCH_RADIX_NUM_UNROLLING * vec_len)
    elem_num_limit = (GCU_VDMEM_SIZE // _TOPK_BPE) // vec_len * vec_len
    if level == 1:
        return tiling_num <= elem_num_limit
    if level == 2:
        return tiling_num <= elem_num_limit * NUM_THREADS
    if level == 3:
        return tiling_num <= elem_num_limit * NUM_THREADS * NUM_BLOCKS
    raise ValueError(f"level must be 1, 2 or 3, got {level}")


def _per_row_radix_level(n_rows: int, n_cols: int) -> int:
    """Cooperation level topsop would run a RADIX_BATCH_SPLIT shape at.

    Mirrors topsop, including its speed-only promotion: when the row still
    fits LEVEL 1 but there are fewer rows than blocks and the row is at least
    256 fp32 elements, LEVEL 2 is preferred. This port does not run what this
    returns; it only reports what topsop would do.

    0 = no level can hold the row (topsop falls back to streaming RADIX),
    1 = one SIP per row,
    2 = one block's 6 SIPs per row,
    3 = all 24 SIPs per row.
    All of them keep the same <<<4, 6>>> launch; only the number of SIPs that
    cooperate on a row changes (see _batch_radix_is_support).
    """
    if _batch_radix_is_support(n_cols, 1):
        level = 1
    elif _batch_radix_is_support(n_cols, 2):
        level = 2
    elif _batch_radix_is_support(n_cols, 3):
        level = 3
    else:
        return 0
    if (level == 1 and n_rows <= NUM_BLOCKS and n_cols >= 256
            and _batch_radix_is_support(n_cols, 2)):
        level = 2
    return level


def topk_tuner(n_rows: int, n_cols: int, k: int) -> int:
    """TopkPattern topsop would pick for this shape (see TopkPattern)."""
    if (n_rows >= DTE_SLICE_MAX_INPUT_DIM_SIZE
            or n_cols >= DTE_SLICE_MAX_INPUT_DIM_SIZE):
        return TopkPattern.RADIX

    if k <= 10:
        if _ktop1_shortcut_is_support(n_rows, n_cols, k):
            return TopkPattern.KTOP1_SHORTCUT
        return TopkPattern.KTOP1_AXIS_SPLIT

    if (_batch_radix_is_support(n_cols, 1)
            or _batch_radix_is_support(n_cols, 2)
            or _batch_radix_is_support(n_cols, 3)):
        return TopkPattern.RADIX_BATCH_SPLIT
    return TopkPattern.RADIX


def _ktop1_shortcut_is_support(n_rows: int, n_cols: int, k: int) -> bool:
    """Admission window of topsop's lane-insert quick path (KTOP1_SHORTCUT).

    The kernel behind it is admitted for k in [3, 10], innermost axis >= 8192,
    and a row count below the per-k bound table (that bound is the row count
    past which the shortcut no longer wins). Every other k <= 10 shape falls
    back to KTOP1_AXIS_SPLIT.

    How it works: a row is cut into segments spread over the SIPs; each
    worker streams its segment once and keeps a small sorted candidate list
    per vector lane, inserting every new element with a fixed-length
    compare-exchange chain (L = 8 slots for k <= 8, 10 above) and never
    talking to other lanes. Each segment then emits its k best, and a second
    stage merges segs * k candidates into the row's final k. The scan touches
    every element exactly once and its cost barely depends on k.

    Why that is the performance opportunity: this file's radix kernel is one
    row per SIP, so it leaves SIPs idle whenever there are fewer rows than
    SIPs, and it always spends two full passes over the row (histogram +
    reclaim). The shortcut splits a single row across many workers and needs
    one pass, so it is the better fit exactly in the corner described here:
    few rows, long axis, small k.
    """
    if not (3 <= k <= 10):          # topsop: KTOP1_SHORTCUT_MIN_K .. MAXK
        return False
    if n_cols < 8192:               # topsop: KTOP1_SHORTCUT_MIN_AXIS
        return False
    # topsop: row-bound tables, keyed by axis <= 32768 / axis > 32768.
    row_bound = (48, 48, 48, 88, 88, 96, 96, 96) if n_cols <= 32768 \
        else (20, 24, 32, 48, 48, 48, 48, 96)
    if n_rows <= 0 or n_rows > row_bound[k - 3]:
        return False
    workers = (NUM_BLOCKS * NUM_THREADS
               * MAX_THREAD_NUM * KTOP1_SHORTCUT_SUBTHREAD_NUM)
    return workers >= n_rows


def _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k):
    """One SIP per row, persistent over rows, no cross-SIP sync.

    Each program owns one row: the tle-raw region receives
    row_base = program_id * BLOCK_SIZE and row_stride = num_programs *
    BLOCK_SIZE (BLOCK_SIZE = 1) and walks its own strided row set, so one
    launch with grid = min(SIP_COUNT, M) blocks of a single thread (4
    subthreads each) covers all M rows.
    """
    if n_cols > L1_MAX_COLS:
        raise ValueError(
            f"N={n_cols} exceeds the one-SIP L1 capacity N<={L1_MAX_COLS}")
    grid_size = min(MAX_GRID, n_rows)
    topk_mradix_l1_launch[(grid_size,)](
        x, y_vals, y_idx,
        n_rows, n_cols, k,
        BLOCK_SIZE=1,          # type: ignore[arg-type]
        num_warps=NUM_WARPS,   # type: ignore[call-arg]
        num_stages=1,          # type: ignore[call-arg]
    )
    return y_vals, y_idx


def tle_raw_topk_mradix(
    x: torch.Tensor,
    k: int,
    out_vals: "torch.Tensor | None" = None,
    out_idx:  "torch.Tensor | None" = None,
):
    """Top-K selection via the TLE-RAW MRADIX radix-select kernel.

    topk_tuner() picks the TopkPattern from shape/geometry (the decision
    topsop makes), then dispatch maps it to a launch:
      - KTOP1 family (KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT / KTOP1_BATCH_SPLIT,
        k <= 10): not implemented here, so these shapes run the radix kernel
        below. That is the better arm for the short-axis shapes it targets and
        the slower one for the KTOP1_SHORTCUT corner (N >= 8192); --bench
        prints the measured ratios.
      - RADIX_BATCH_SPLIT: persistent kernel, grid = min(SIP_COUNT, M) = 24 hw
        threads (one per SIP); no inter-thread sync. Only the LEVEL=1 layout
        (one SIP per row) is implemented.
      - every other pattern: rejected below (streaming RADIX, HEAP/QUICK/
        BITONIC/SCAN/SORT, and the LEVEL>=2 cooperation shapes).
    """
    if x.ndim != 2:
        raise ValueError(f"input must be 2D (M, N), got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"the edsl kernel is fp32-only, got {x.dtype}")
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")
    # Capacity envelope of the edsl kernel (see the "Capacity" note in the
    # header of 04-topk-mradix.tops):
    #   - K <= TOPK_K_MAX (640): the reclaim staging in the 16KB ws buffer has
    #     WS_STAGE_K_MAX slots for each of the gt and eq streams.
    #   - N <= L1_MAX_COLS (65536): the kernel caches the whole row in the l1
    #     buffer this launcher allocates.
    # Both are checked before dispatch, so an oversized shape fails with one
    # message instead of overflowing VDMEM.
    if k > TOPK_K_MAX:
        raise ValueError(
            f"k={k} exceeds the edsl reclaim-staging limit k<={TOPK_K_MAX}")

    if out_vals is None:
        y_vals = torch.empty((n_rows, k), device=x.device, dtype=x.dtype)
    else:
        y_vals = out_vals
    if out_idx is None:
        y_idx = torch.empty((n_rows, k), device=x.device, dtype=torch.int32)
    else:
        y_idx = out_idx

    # --------------------------- pattern dispatch ---------------------------
    pattern = topk_tuner(n_rows, n_cols, k)
    if pattern in _KTOP1_PATTERNS:
        # k <= 10: topsop runs its KTOP1 kernels (lane-insert shortcut, or the
        # axis-split variant when the shortcut window does not admit the
        # shape). This port has no KTOP1 implementation, so every k <= 10
        # shape runs the radix kernel; the row must still fit one SIP's VDMEM,
        # which _launch_batch_radix checks.
        return _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k)
    if pattern == TopkPattern.RADIX_BATCH_SPLIT:
        # topsop refines this label into three cooperation levels: 1 is one SIP
        # per row, 2 a block's threads per row, 3 all blocks per row (all three
        # keep the <<<4, 6>>> launch). Levels 2/3 exist to handle a row that
        # does not fit one SIP's VDMEM -- the row is sliced across the SIPs and
        # the histogram plus the candidates live in L3 workspace behind a
        # block-scope (2) or grid-scope (3) barrier -- and topsop also promotes
        # a few-rows/wide-row shape from 1 to 2 purely for speed.
        #
        # This port implements LEVEL 1 and always runs it: the promoted path
        # needs a cross-SIP barrier, and the TLE-RAW grid-scope barrier costs
        # ~7x topsop's block-scope one, so the promoted launch measured slower
        # than one SIP per row on every shape tried. A row that needs LEVEL 2/3
        # for *capacity* has no kernel here, hence the check below.
        if not _batch_radix_is_support(n_cols, 1):
            raise ValueError(
                f"N={n_cols} does not fit one SIP's VDMEM: topsop runs this "
                f"shape at LEVEL={_per_row_radix_level(n_rows, n_cols)}; this "
                f"port implements LEVEL=1 only")
        return _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k)
    # Every other topsop pattern is recognized (see TopkPattern) but has no
    # TLE-RAW implementation: RADIX (streaming: no level can hold the row, so
    # it stays in HBM and is re-read once per 5-bit digit - up to 7 for fp32,
    # early-exiting once the k-th bin resolves - again through a per-SIP
    # VDMEM tile, with the histogram and the reclaimed candidates in global
    # workspace; also reached when a dim passes the DTE slice limit),
    # HEAP / QUICK / BITONIC / SCAN / SORT.
    raise ValueError(
        f"TopkPattern {_PATTERN_NAMES.get(pattern, pattern)} not supported "
        f"for shape ({n_rows}, {n_cols}), k={k}")


def _assert_close_vals(a: torch.Tensor, b: torch.Tensor, tag: str):
    """Compare two tiles with torch.testing.assert_close, on the CPU.

    Both operands are moved to the CPU before comparing: a device-side
    torch.all/torch.allclose bool reduction is avoided on gcu400, where it has
    been observed to be followed by a "Sip exception" KMD abort on the next
    cooperative launch.
    """
    if a.shape != b.shape:
        raise AssertionError(f"{tag}: shape mismatch {a.shape} vs {b.shape}")
    torch.testing.assert_close(a.to("cpu"), b.to("cpu"),
                               rtol=0.0, atol=1e-3, equal_nan=True, msg=tag)


def _check_topk(x: torch.Tensor, y_vals: torch.Tensor, y_idx: torch.Tensor,
                tag: str):
    """Verify one tle-raw result against torch.topk, order-insensitively.

    The returned order is unspecified (reclaimed order), so the values are
    compared as sorted multisets. The index stream is pinned separately: every
    returned index must carry exactly the value returned next to it.
    """
    k = y_vals.shape[1]
    t_vals, _ = torch.topk(x, k, dim=1, sorted=False)
    _assert_close_vals(
        torch.sort(y_vals, dim=1, descending=True).values,
        torch.sort(t_vals, dim=1, descending=True).values, f"values {tag}")
    _assert_close_vals(x.gather(1, y_idx.to(torch.int64)), y_vals,
                       f"indices {tag}")


def run_correctness_on(x: torch.Tensor, k: int, tag: str):
    """Run one tle-raw topk and verify it against torch.topk."""
    y_vals, y_idx = tle_raw_topk_mradix(x, k)
    _check_topk(x, y_vals, y_idx, tag)
    print(f"MRADIX TopK correctness PASSED ({tag})")


def run_correctness(m: int, n: int, k: int):
    """Uniform random floats."""
    torch.manual_seed(42)
    run_correctness_on(torch.rand((m, n), device=DEVICE, dtype=torch.float32),
                       k, f"M={m} N={n} K={k}")


def run_benchmark(m: int, n: int, k: int):
    """Quick manual benchmark (kept for ad-hoc debugging).

    The canonical performance harness is the perf_report benchmark below
    (test_benchmark_topk_*), whose numbers cross-check against topsprof
    --print-gcu-trace kernel times.
    """
    torch.manual_seed(42)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)

    def fn_mradix():
        return tle_raw_topk_mradix(x, k)

    def fn_torch():
        return torch.topk(x, k, dim=1, sorted=False)

    ms_tle_raw = triton.testing.do_bench(fn_mradix, warmup=50, rep=200)
    ms_torch   = triton.testing.do_bench(fn_torch,  warmup=50, rep=200)
    # ratio = tle-raw / torch (<1: this kernel is faster) and speedup = its
    # reciprocal, torch / tle-raw (>1: this kernel is faster).
    ratio = ms_tle_raw / ms_torch
    print(f"[M={m:4d}, N={n:5d}, K={k:3d}]  "
          f"torch: {ms_torch:.3f} ms  |  "
          f"tle-raw: {ms_tle_raw:.3f} ms  |  "
          f"ratio: {ratio:.2f}x  |  "
          f"speedup: {1.0 / ratio:.2f}x")


# =============================================================================
# Performance Tests (perf_report + do_bench(quantiles))
# =============================================================================
#
# Measurement methodology notes (validated against topsprof --print-gcu-trace
# kernel times, which are the ground truth):
#   - do_bench times HOST-side elapsed time per fn() call = pure kernel time
#     (topsprof) + a fixed ~7 us launch/host overhead. Measured pairs:
#     (128,32768,256) 133.0 us kernel vs 140.0 us do_bench (+7.0 us);
#     (8,4096,32) 9.4 us kernel vs 17.0 us do_bench (+7.6 us). So for kernels
#     >> overhead the two agree, and for tiny kernels do_bench is
#     launch-overhead-dominated - read those rows as end-to-end cost, not
#     kernel cost.
#   - do_bench's built-in n_warmup/n_repeat estimation plus its 256MB L2
#     cache-clear between reps make single-call timing stable (p20/p80 spread
#     < 3% here), so no extra inner loop is needed. A loop of several calls per
#     timed region is avoided on purpose: it measures steady-state throughput
#     instead of per-invocation kernel time, so the numbers stop matching
#     topsprof.
#   - Correctness inside the benchmark goes through _check_topk, which compares
#     on the CPU (see _assert_close_vals).

# Main benchmark set, grouped by the topsop-aligned TopkPattern each shape
# routes to (verified against topk_tuner) - the 3 supported groups plus one
# output-alignment group:
#   KTOP1 (k <= 10): KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT / KTOP1_BATCH_SPLIT
#     share one dispatch arm that runs the radix kernel; the label only
#     records which topsop pattern the shape maps to.
#     (1,16384,8) sits inside the ktop1_shortcut admission window (k=8,
#     N=16384, 1 <= row_bound 96) so the SHORTCUT arm is exercised too.
#   RADIX_BATCH_SPLIT (k > 10): all k > 10 shapes that fit one SIP.
#   LARGE_ROW: the wide-row shapes (N=16384), kept as a separate table
#     because they are the closest thing to a bandwidth-bound regime.
#   UNALIGNED_K: K not a multiple of VEC_LEN (128 words), so the output rows
#     are 512B-misaligned except where r*K*4 % 512 == 0. Same pattern as
#     RADIX_BATCH_SPLIT; split out because it is the S5 writeback regime
#     (element-aligned VUI store + scalar tail) and the row stride decides
#     which rows take the misaligned store.
BENCH_CASES_KTOP1 = [
    (1,   1024,    8),
    (8,   1024,    8),
    (16,  1024,    8),
    (64,  128,     8),
    (1,   16384,   8),   # KTOP1_SHORTCUT window
]
BENCH_CASES_L1 = [
    # N=4096: small-batch controls (M below SIP_COUNT, so a single wave).
    (1,   4096,   32),
    (1,   4096,   64),
    (4,   4096,   32),
    (4,   4096,   64),
    (8,   4096,   32),
    (8,   4096,   64),
    # Larger M and the widest supported row (N=32768, K=256).
    (64,  1024,   32),
    (64,  8192,  128),
    (128, 32768, 256),
]
# Wide-row group: one SIP still owns a whole row, so these are the shapes
# where the kernel is closest to VDMEM-bandwidth bound.
BENCH_CASES_LARGE_ROW = [
    (1,   16384,  32),
    (1,   16384,  64),
    (4,   16384,  32),
    (4,   16384,  64),
    (8,   16384,  32),
    (8,   16384,  64),
]
# Unaligned-K group: same 8 shapes as the correctness group of that name, so
# a regression shows up in both. Row strides are 400B (K=100), 516B (129),
# 544B (136), 800B (200), 928B (232), 1200B (300), 1552B (388) and 2080B
# (520); K=100 stays on the K < VEC_LEN deslice writeback.
BENCH_CASES_UNALIGNED_K = [
    (32,  8192,  100),
    (32,  8192,  129),
    (32,  8192,  136),
    (32,  8192,  200),
    (16,  4096,  232),
    (32,  8192,  300),
    (16,  8192,  388),
    (32,  8192,  520),
]


def _bench_one(provider, m, n, k):
    """Benchmark one (M, N, K) x provider; returns (median, p20, p80) in ms.

    The input tensor is allocated fresh for every (shape, provider) pair, so a
    timing does not depend on which shape ran before it.
    """
    torch.manual_seed(42)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        return triton.testing.do_bench(
            lambda: torch.topk(x, k, dim=1, sorted=False), quantiles=quantiles)

    # provider == "tle-raw": verify correctness once per (shape, provider)
    # before timing (host-side comparison, see _assert_close_vals).
    y_vals, y_idx = tle_raw_topk_mradix(x, k)
    _check_topk(x, y_vals, y_idx, f"bench M={m} N={n} K={k}")

    return triton.testing.do_bench(
        lambda: tle_raw_topk_mradix(x, k), quantiles=quantiles)


def _make_topk_benchmark(plot_name, cases):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=list(cases),
            line_arg='provider',
            line_vals=['torch', 'tle-raw'],
            line_names=["Torch", "TLE-RAW"],
            styles=[('green', '-'), ('blue', '-')],
            ylabel="us",
            plot_name=plot_name,
            args={},
        ))(
        lambda M, N, K, provider: tuple(
            us * 1000.0 for us in _bench_one(provider, M, N, K)))


class TestTopkPerf:
    """Performance benchmark for TLE-RAW MRADIX topk (topsprof-aligned)."""

    @pytest.fixture(autouse=True)
    def setup_benchmark(self, tmp_path='./perf_results'):
        # matplotlib saves the PNG before benchmark.run() creates save_path,
        # so the directory must already exist here.
        os.makedirs(str(tmp_path), exist_ok=True)
        self._save_path = str(tmp_path)

    def test_benchmark_topk_ktop1(self):
        """KTOP1 group (k <= 10; KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT /
        KTOP1_BATCH_SPLIT all run the L1 radix edsl, see dispatch)."""
        benchmark = _make_topk_benchmark(
            "gcu400-tle-raw-topk-mradix-ktop1", BENCH_CASES_KTOP1)
        benchmark.run(show_plots=True, print_data=True,
                      save_path=self._save_path)

    def test_benchmark_topk_l1(self):
        """RADIX_BATCH_SPLIT L1 group."""
        benchmark = _make_topk_benchmark(
            "gcu400-tle-raw-topk-mradix-l1", BENCH_CASES_L1)
        benchmark.run(show_plots=True, print_data=True,
                      save_path=self._save_path)

    def test_benchmark_topk_large_row(self):
        """Wide-row (N=16384) group."""
        benchmark = _make_topk_benchmark(
            "gcu400-tle-raw-topk-mradix-large-row", BENCH_CASES_LARGE_ROW)
        benchmark.run(show_plots=True, print_data=True,
                      save_path=self._save_path)

    def test_benchmark_topk_unaligned_k(self):
        """Unaligned-K group (K % 128 != 0, misaligned output rows)."""
        benchmark = _make_topk_benchmark(
            "gcu400-tle-raw-topk-mradix-unaligned-k", BENCH_CASES_UNALIGNED_K)
        benchmark.run(show_plots=True, print_data=True,
                      save_path=self._save_path)


def run_correctness_mixed(m: int, n: int, k: int):
    """Mixed positive/negative values (randn)."""
    torch.manual_seed(123)
    run_correctness_on(
        torch.randn((m, n), device=DEVICE, dtype=torch.float32), k,
        f"M={m} N={n} K={k} mixed +/-")


def run_correctness_ties(m: int, n: int, k: int):
    """Duplicate values, i.e. ties at the k-th value (the split path).

    Few distinct levels make n_ge > K on most rows; the all-equal tensor is the
    extreme (n_ge == N, n_gt == 0). Which duplicate ends up in the output is
    unspecified, so _check_topk compares value multisets.
    """
    torch.manual_seed(7)
    run_correctness_on(
        torch.randint(0, 4, (m, n)).to(torch.float32).to(DEVICE), k,
        f"M={m} N={n} K={k} ties/4 levels")
    run_correctness_on(
        torch.full((m, n), 0.25, dtype=torch.float32, device=DEVICE), k,
        f"M={m} N={n} K={k} ties/all-equal")


if __name__ == "__main__":
    if "--perf" in sys.argv:
        # perf_report tables + CSV/PNG under ./perf_results/, cross-checkable
        # with topsprof.
        pytest.main([__file__, "-v", "-k", "benchmark", "--noconftest"])
        sys.exit(0)

    print(f"SIP_COUNT={SIP_COUNT}, NUM_WARPS={NUM_WARPS}, MAX_GRID={MAX_GRID}")
    print()

    # Correctness sets: the two TopkPattern groups topk_tuner routes k > 10 /
    # k <= 10 shapes to, plus the edge-case groups (unaligned K, mixed sign,
    # duplicate values, wide rows) that stress one stage each. Every shape runs
    # the same radix kernel, so the value/index check is the only thing that
    # differs between the groups.
    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (KTOP1 group, k<=10)")
    print("=" * 60)
    for m, n, k in [(  1,    8,  3),
                    (  2,   16,  4),
                    (  4,  128,  8),
                    (  8,  256,  4),
                    (  8, 1024,  8),
                    (  1, 4096,  10),
                    ( 16,  256,  4),
                    ( 32, 1024,  8),
                    ( 64,  128,  8),
                    (128,  256,  4),
                    (  1, 16384, 8)]:   # KTOP1_SHORTCUT window
        run_correctness(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (RADIX_BATCH_SPLIT group, k>10)")
    print("=" * 60)
    for m, n, k in [(  1, 4096,  32),
                    (  1, 4096,  64),
                    (  4, 4096,  32),
                    (  8, 4096,  64),
                    (  2, 8192,  16),
                    (  8, 8192, 128),
                    ( 64, 8192, 128),
                    (128, 32768, 256)]:
        run_correctness(m, n, k)
    print()

    # K not a multiple of VEC_LEN (128 words) makes the output rows 512B
    # misaligned except for the rows where r*K*4 % 512 == 0, which is what
    # exercises S5's element-aligned VUI store and its scalar tail. Row
    # strides: 400B (K=100), 516B (129), 544B (136), 800B (200), 928B (232),
    # 1200B (300), 1552B (388), 2080B (520). K=100 (400B) stays on the
    # K < 128 deslice path with a misaligned row start.
    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (unaligned-K group, K % 128 != 0)")
    print("=" * 60)
    for m, n, k in [( 32, 8192, 100),
                    ( 32, 8192, 129),
                    ( 32, 8192, 136),
                    ( 32, 8192, 200),
                    ( 16, 4096, 232),
                    ( 32, 8192, 300),
                    ( 16, 8192, 388),
                    ( 32, 8192, 520)]:
        run_correctness(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (mixed +/- randn)")
    print("=" * 60)
    for m, n, k in [(1, 8, 3), (4, 128, 8), (8, 1024, 16), (1, 4096, 32)]:
        run_correctness_mixed(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (duplicate values, ties at k)")
    print("=" * 60)
    for m, n, k in [(1, 256, 8),
                    (8, 1024, 32),
                    (4, 4096, 128),
                    (2, 8192, 512)]:
        run_correctness_ties(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (large-row group)")
    print("=" * 60)
    for m, n, k in [(1, 16384,  32),
                    (1, 16384,  64),
                    (4, 16384,  32),
                    (4, 16384,  64),
                    (8, 16384,  32),
                    (8, 16384,  64)]:
        run_correctness(m, n, k)
    print()

    print("\nAll MRADIX TopK correctness tests PASSED")

    # Benchmarks are opt-in: the default entry point is what the tops-script
    # regression test runs, and it must stay correctness-only.
    if "--bench" in sys.argv:
        print()
        print("=" * 60)
        print("MRADIX TopK - Performance Benchmarks (KTOP1 group, k<=10)")
        print("=" * 60)
        for m, n, k in BENCH_CASES_KTOP1:
            run_benchmark(m, n, k)

        print()
        print("=" * 60)
        print("MRADIX TopK - Performance Benchmarks (RADIX_BATCH_SPLIT group)")
        print("=" * 60)
        for m, n, k in BENCH_CASES_L1:
            run_benchmark(m, n, k)

        print()
        print("=" * 60)
        print("MRADIX TopK - Performance Benchmarks (large-row group)")
        print("=" * 60)
        for m, n, k in BENCH_CASES_LARGE_ROW:
            run_benchmark(m, n, k)

        print()
        print("=" * 60)
        print("MRADIX TopK - Performance Benchmarks (unaligned-K group)")
        print("=" * 60)
        for m, n, k in BENCH_CASES_UNALIGNED_K:
            run_benchmark(m, n, k)
