"""
TLE-Struct + TLE-Raw Backend: TopK with HW MRADIX Acceleration
==============================================================

One hw thread per row: the geometry of topsop's LEVEL=1 TopK. Verified live
(OPLIB TopsFlame logs + device props): topsop's NORMAL torch.topk launches
<<<numBlocks=multiProcessorCount=4, numThreads=maxThreadsPerBlock=6>>> = 24
hw threads, ONE per SIP, each with TOPK_RADIX_SUBTHREAD_NUM=4 subthreads and
GROUP_NUM=1 (group_num=2 exists only for TOPK_PERROW_PREFILL/INDEXER with
batch>=48). topsop's LEVEL=1 capacity is the full 896KB
(GCU_VDMEM_SIZE/bpe = 229376 fp32); this launcher instead caches the row in
the VDMEM window it allocates and caps at L1_MAX_COLS = 65536.

This is 04-topk-mradix.py with every data movement moved to TLE-Struct and
the per-row compute left in TLE-Raw. The same kernel also exists as one raw
region with its own row loop (04-topk-mradix.py + .tops); keeping both lets
the boundary be measured against the standalone version.

Design -- S1..S5, and the side of the Struct/Raw boundary that runs each one:
  - S1 load (Struct): tle.gpu.copy stages one row into a VDMEM window, masked
    to offs < N and padded with -inf, so the tail lanes can never be
    selected.
  - S1 convert (raw): all 4 subthreads turn the staged fp32 row into
    order-preserving uint32 keys, in place.
  - S2 histogram (raw): all 7 digit passes on HW MRADIX (5-bit digit, MSB
    first), sub0 only -- MRADIX is a per-SIP shared unit. find_k_bin then
    walks the 32 bins to the digit holding the k-th largest key.
  - S3 reclaim (raw): ge counts, exclusive prefix and compression table,
    then the lane scatter into staging. Two paths: the common case
    (n_ge == K, unique floats) runs one ge count + ge scatter; ties at the
    k-th value (n_ge > K) recount with '>' and split-scatter gt then eq,
    truncated at K.
  - S4 sort (raw): no-op -- the k lanes come back in reclaimed order, which
    is torch.topk(sorted=False).
  - S5 writeback (Struct): the k staged lanes are read through
    tle.gpu.local_ptr and stored as one masked K-wide tile, so no DTE is
    issued on either side of the boundary.
  - Row loop (Struct, persistent): grid = min(SIP_COUNT, M) blocks of one hw
    thread, no inter-thread synchronization. Each program walks its strided
    row set with row = row_base + r * row_stride and calls the raw region
    once per row.

  Why the split falls here: the row loop, the row copy and the k-element
  writeback are ordinary Triton work, so expressing them as tle.gpu.copy /
  tl.load / tl.store costs nothing and keeps the raw region free of data
  movement; the raw region keeps only what needs tcle and MRADIX. Both VDMEM
  windows are declared once, on the Struct side, and handed over as pointers
  (see the buffer-ownership note in topk_mradix_l1_launch).

Launch: grid = min(SIP_COUNT, M), no cluster dims (one hw thread per SIP,
matching topsop's NORMAL-topk geometry); num_warps=4 = topsop
TOPK_RADIX_SUBTHREAD_NUM subthreads. Because rows are walked serially inside
each hw thread, the wall clock is ceil(M/SIP_COUNT) * T_row, not M/SIP_COUNT
* T_row. T_row here is the standalone kernel's T_row plus the row copy, which
no longer overlaps the compute; the measured model and its consequences are
in the header of 04-topk-mradix-struct-raw-persistent.tops.

Out of scope: topsop LEVEL>=2 (several SIPs cooperating on one row), KTOP1
(k <= 10 comparison-tree kernels) and streaming RADIX; see the limits list in
04-topk-mradix-struct-raw-persistent.tops and topk_tuner().

Entry points (python 04-topk-mradix-struct-raw-persistent.py):
  (default)  correctness suite only (what the tops-script regression runs)
  --bench    correctness + the inline benchmark tables
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
SIP_COUNT = torch.gcu.get_device_properties(DEVICE.index).sip_count
NUM_BLOCKS = torch.gcu.get_device_properties(DEVICE.index).multi_processor_count
NUM_THREADS = torch.gcu.get_device_properties(DEVICE.index).max_threads_per_multi_processor
MAX_GRID = SIP_COUNT
NUM_WARPS = 4  # No support topk_per_row_hetero which use 8 warps.

GCU_VDMEM_SIZE = 0xE0000  # 896KB per SIP
TCLE_SINGLE_VEC_BYTES = 512  # OACC = 512B
DTE_SLICE_MAX_INPUT_DIM_SIZE = 0x1000000  # 2^24
BATCH_RADIX_NUM_UNROLLING = 16
MAX_THREAD_NUM = 2
KTOP1_SHORTCUT_SUBTHREAD_NUM = 8
_TOPK_BPE = 4  # the raw region is fp32-only.

# LEVEL=1 radix capacity. The SIP's whole VDMEM is 896KB (229376 fp32 elems),
# but this launcher caches the whole row in the window it allocates
# (L1_MAX_COLS = 65536 elems = 256KB) and keeps a 16KB reclaim workspace /
# staging arena next to it, i.e. 272KB of 896KB. 65536 is the contract: rows
# above it are rejected here instead of silently overflowing VDMEM. It is also
# the largest row the L1 radix-select is competitive on (every digit pass
# rescans the whole row).
L1_MAX_COLS = 65536
# Widest k the reclaim staging supports (WS_STAGE_K_MAX in the raw region).
# The ws buffer is L1_WS_ELEMS = 4096 uint32 elements and its layout is
#   [0, 2560)            counts / prefixes / compression table
#                        (5 * WS_FC_AL_MAX, fc_al <= 512 at N = 65536)
#   [2560, 2560 + 2*K)   stage_v[K], stage_i[K]
#   [4093, 4096)         values published across subthreads during reclaim:
#                        n_gt (split path), comp_cnt, and desired which is then
#                        reused for n_ge
# so staging must stop below the publish words: 2*K <= 4096 - 2560 - 3 = 1533
# -> K <= 766. stage_i also has to start 512B-aligned because the S5 read takes
# both streams as whole 128-lane tiles, i.e. K must be a multiple of 128 words
# -> K_MAX = 640.
# To support a larger k, enlarge the ws allocation (L1_WS_ELEMS here and
# WS_SIZE in the raw region; the SIP has ~624KB of VDMEM free at 272KB used), keep
# WS_STAGE_OFF, and set WS_STAGE_K_MAX / TOPK_K_MAX to the same
# (WS_SIZE - WS_STAGE_OFF - 3) // 2 rounded down to a multiple of 128. Moving
# the staging to HBM is not an alternative: a global scatter from the
# subthreads poisons the device (see the S3 note in the raw region), and a k-sized
# global round trip per row is dearer than the 16KB window it replaces.
TOPK_K_MAX = 640
# The VDMEM windows this launcher hands to the raw region. The row buffer is
# (1, BLOCK_N) with BLOCK_N = max(128, next_pow2(N)) <= L1_MAX_COLS, i.e.
# <= 256KB of the SIP's 896KB VDMEM; the reclaim workspace is the fixed 16KB ws
# buffer. Both must be power-of-two shapes for tle.gpu.alloc. The raw region is
# written against element counts, not bytes: the row buffer holds one row and
# then its 32-bit sort keys (converted in place), which is why the size stays
# the same for a narrower input dtype (bf16/fp16).
#   L1_WS_ELEMS   == WS_SIZE   (04-topk-mradix-struct-raw-persistent.tops; which
#                               also pins WS_STAGE_OFF / WS_STAGE_K_MAX, so keep
#                               the three in sync)
L1_WS_ELEMS = tl.constexpr(4096)
# Staging layout inside ws (words), see the ws comment: the raw region scatters
# the k selected lanes here and the S5 writeback below reads them back.
WS_STAGE_OFF = tl.constexpr(2560)  # == WS_STAGE_OFF in the raw region
WS_STAGE_K_MAX = tl.constexpr(640)  # == WS_STAGE_K_MAX (== TOPK_K_MAX)
# Width of the masked S5 read tile. One power-of-two tile covers every k
# (k <= 640) without specializing the kernel per k; a wider tile only adds
# masked-off lanes.
WS_S5_BLOCK_K = tl.constexpr(1024)


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


# The raw region: 04-topk-mradix-struct-raw-persistent.tops compiled as a
# standalone device function. deferred=True makes the call site a dsl_region
# that is materialized at the end of the pipeline instead of being inlined at
# the call site, so the region can sit inside the row loop and is traced once,
# not once per iteration.
@dialect(name="tops", file=Path(__file__).parent / "04-topk-mradix-struct-raw-persistent.tops",
         extern_func_name="topk_mradix_l1_kernel", deferred=True)
def edsl_topk_mradix_l1(*args, **kwargs):
    ...


# The Struct side of the pipeline: one launch per shape, one raw call per row.
# The loop lives here (not in the region) because this side owns the row buffer
# and the staging arena, and because the row mapping is then plain Triton
# arithmetic instead of a second copy of it inside the region.
@triton.jit(do_not_specialize=["M", "N", "K"])
def topk_mradix_l1_launch(
    input_ptr,
    output_vals_ptr,
    output_idx_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # No scratch argument: the level-1 kernel keeps the histogram in MRADIX's
    # own register file and the reclaim workspace / staging in l1 VDMEM.
    #
    # Buffer ownership: both VDMEM windows are allocated HERE, on the
    # TLE-Struct side, and handed to the raw region as plain pointers (see the
    # OWNERSHIP note at its entry in
    # 04-topk-mradix-struct-raw-persistent.tops). The raw region must not
    # declare them __local__ as well: both allocators start at VDMEM offset 0
    # and neither side knows about the other, so raw static buffers would
    # silently overlap this window (measured: the overlapping layout fails on
    # every shape and passes once shifted by one raw buffer). Handing the
    # pointers over costs ~0.37us per launch, which the persistent loop
    # amortizes over all of this hw thread's rows.
    #
    # S1 load: BLOCK_N = max(128, next_pow2(N)) lanes, so the copy extent is a
    # power of two (tl.arange) and the mask/pad only covers the row's own tail
    # plus the 128-lane vector chunk S1 convert reads. The buffer is a 2-D
    # (1, BLOCK_N) window so the DTE destination is a slot subview -- the form
    # that keeps the copy a real slice -- and it grows to (2, BLOCK_N) without
    # touching this code if the row copy is ever made asynchronous.
    #
    # S5 writeback: the raw region leaves the k selected lanes in staging, at
    # ws[WS_STAGE_OFF ..] (values) and ws[WS_STAGE_OFF + WS_STAGE_K_MAX ..]
    # (indices), as 32-bit words; this side reads them through
    # tle.gpu.local_ptr and stores one masked K-wide tile per stream. No DTE is
    # issued on either side of the boundary.
    row_base = tl.program_id(0) * BLOCK_SIZE
    row_stride = tl.num_programs(0) * BLOCK_SIZE
    # Exact trip count for this hw thread: no tail clamp, so every iteration
    # owns a real row and S5 can store at row*K unconditionally.
    nrows = tl.cdiv(M - row_base, row_stride)
    l1 = tle.gpu.alloc([1, BLOCK_N], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ws = tle.gpu.alloc([L1_WS_ELEMS], dtype=tl.uint32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ws_ptr = tle.gpu.local_ptr(ws, (0, ))
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, WS_S5_BLOCK_K)
    k_mask = offs_k < K
    for r in range(0, nrows):
        row = row_base + r * row_stride
        # S1 load: pad lanes arrive as -inf, which S1 convert maps to
        # 0x007fffff -- below every finite key, so they can never be selected.
        tle.gpu.copy(input_ptr + row * N + offs_n, l1.slot(0), [BLOCK_N], mask=offs_n < N, other=float("-inf"))
        # S1 convert .. S4 over the staged row, in place. Both VDMEM windows
        # are written through their pointers. l1 holds the sort keys and ws
        # holds the published counts and the k-element staging.
        tle_raw.call(edsl_topk_mradix_l1, [tle.gpu.local_ptr(l1, (0, 0)), ws_ptr, N, K], output_indices=[])
        # S5 writeback: both staged streams are 32-bit words, bitcast on the
        # way out. WS_STAGE_K_MAX is the staging stride fixed by the raw region and
        # pinned by TOPK_K_MAX here; a wider read tile only adds masked-off
        # lanes.
        sv = tl.load(tle.gpu.local_ptr(ws, (WS_STAGE_OFF + offs_k, )), mask=k_mask)
        si = tl.load(tle.gpu.local_ptr(ws, (WS_STAGE_OFF + WS_STAGE_K_MAX + offs_k, )), mask=k_mask)
        tl.store(output_vals_ptr + row * K + offs_k, sv.to(tl.float32, bitcast=True), mask=k_mask)
        tl.store(output_idx_ptr + row * K + offs_k, si.to(tl.int32, bitcast=True), mask=k_mask)
        # Staging is reused by the next iteration, whose S3 scatters into it
        # again. That is already ordered by the region's own S1/S2 barriers
        # (no subthread reaches S3 before all of them have passed S1, i.e.
        # before all of them have finished this read), so this barrier is
        # defence in depth. Measured cost as a block-wide barrier per row:
        # ~0.2us/row, 1.2us on 128x32768x256.
        tl.debug_barrier()


def tle_raw_topk_mradix(
    x: torch.Tensor,
    k: int,
    out_vals: "torch.Tensor | None" = None,
    out_idx: "torch.Tensor | None" = None,
):
    """Top-K selection via TLE-Struct staging + TLE-Raw MRADIX radix-select.

    topk_tuner() picks a TopkPattern from shape/geometry (topsop-aligned),
    then dispatch maps it to one kernel:
      - KTOP1 family (KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT / KTOP1_BATCH_SPLIT,
        k <= 10): not ported - these shapes run the LEVEL=1 radix kernel
        below. That is the right arm for the short-axis shapes it targets,
        but not for the KTOP1_SHORTCUT corner (N >= 8192), where topsop's
        lane-insert kernels are faster. Same do_bench harness as the
        benchmark tables at the bottom of this file.
      - RADIX_BATCH_SPLIT: persistent kernel, grid = min(SIP_COUNT, M) =
        24 hw threads (one per SIP - verified topsop NORMAL geometry
        <<<4 blocks, 6 threads>>>, GROUP_NUM=1); no inter-thread sync.
      - every other pattern: asserted unsupported for these shapes (topsop
        LEVEL>=2 cooperation, streaming RADIX, HEAP/QUICK/BITONIC/SCAN/SORT).
    """
    assert x.ndim == 2, "input must be 2D (M, N)"
    n_rows, n_cols = x.shape
    if k > n_cols:
        raise ValueError(f"k={k} must be <= N={n_cols}")
    # Capacity envelope of the raw region (see the "Capacity" note in
    # 04-topk-mradix-struct-raw-persistent.tops):
    #   - K <= WS_STAGE_K_MAX (640): the reclaim staging in the 16KB ws buffer
    #     has WS_STAGE_K_MAX slots for each of the gt and eq streams.
    #   - N <= L1_MAX_COLS (65536): the kernel caches the whole row in the l1
    #     buffer we allocate; enforced at the dispatch arms below.
    # The K check sits here, before dispatch, so both arms (k <= 10 and
    # k > 10) reject an oversized K with one message instead of overflowing
    # VDMEM.
    if k > TOPK_K_MAX:
        raise ValueError(f"k={k} exceeds the reclaim-staging limit k<={TOPK_K_MAX}")

    if out_vals is None:
        y_vals = torch.empty((n_rows, k), device=x.device, dtype=x.dtype)
    else:
        y_vals = out_vals
    if out_idx is None:
        y_idx = torch.empty((n_rows, k), device=x.device, dtype=torch.int32)
    else:
        y_idx = out_idx

    # ---------------------------- pattern dispatch ----------------------------
    pattern = topk_tuner(n_rows, n_cols, k)
    if pattern in (TopkPattern.KTOP1_SHORTCUT, TopkPattern.KTOP1_AXIS_SPLIT, TopkPattern.KTOP1_BATCH_SPLIT):
        # K <= 10: topsop runs its KTOP1 kernels (lane-insert shortcut, or the
        # axis-split variant when the shortcut window does not admit the
        # shape). This port has no KTOP1 implementation, so every k <= 10
        # shape runs the LEVEL=1 radix kernel below. Radix is the better arm
        # for the short-axis shapes it targets, but it is slower than
        # topsop's KTOP1_SHORTCUT for N >= 8192 (same do_bench harness as the
        # benchmark tables at the bottom of this file). The row must still fit
        # one SIP's VDMEM, which the assert below guards.
        assert n_cols <= L1_MAX_COLS, (f"KTOP1 (k<=10) shapes need N={n_cols} <= L1 capacity "
                                       f"{L1_MAX_COLS}; larger rows route to RADIX_BATCH_SPLIT in topsop")
        return _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k)
    if pattern == TopkPattern.RADIX_BATCH_SPLIT:
        # topsop refines this label into three cooperation levels: 1 is one SIP
        # per row, 2 a block's threads per row, 3 all blocks per row. All three
        # keep the <<<4, 6>>> launch (4 blocks x 6 SIPs); the level only picks
        # how many of those SIPs cooperate on a row. Levels 2/3 are how it
        # handles a row that does not fit one SIP's VDMEM: the row is sliced
        # across the SIPs, each keeps only its own slice in VDMEM, and the
        # histogram plus each SIP's reclaimed candidates live in L3 workspace
        # (cross-SIP barrier: block scope for 2, grid scope for 3). topsop
        # picks 2 whenever one SIP cannot hold the row, and additionally
        # promotes a few-rows/wide-row shape from 1 to 2 purely for speed.
        #
        # This port implements LEVEL 1 only and always runs it. The promoted
        # path needs a cross-SIP barrier, and under TLE-RAW the grid-scope
        # barrier costs ~7x topsop's block-scope one; the promoted launch
        # measured slower than one SIP per row on every shape tried
        # ((1,8192,64) 1.32x vs 1.79x, (1,4096,32) 1.03x vs 1.68x against
        # torch.topk, where <1 is faster), so a shape promoted only for speed
        # runs LEVEL 1. A row that needs LEVEL 2/3 for *capacity* has no
        # kernel here, hence the assert.
        assert _batch_radix_is_support(n_cols, 1), (f"N={n_cols} does not fit one SIP's VDMEM: topsop runs this shape "
                                                    f"at LEVEL={_per_row_radix_level(n_rows, n_cols)}; this port "
                                                    f"implements LEVEL=1 only")
        return _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k)
    # Every other topsop pattern is recognized (see TopkPattern) but has no
    # TLE-RAW implementation: RADIX (streaming: no level can hold the row, so
    # it stays in HBM and is re-read once per 5-bit digit - up to 7 for fp32,
    # early-exiting once the k-th bin resolves - again through a per-SIP
    # VDMEM tile, with the histogram and the reclaimed candidates in global
    # workspace; also reached when a dim passes the DTE slice limit),
    # HEAP / QUICK / BITONIC / SCAN / SORT.
    assert False, (f"TopkPattern {_PATTERN_NAMES.get(pattern, pattern)} not supported "
                   f"for shape ({n_rows}, {n_cols}), k={k}")


class TopkPattern:
    """Mirror of topsop's TopkPattern enum.

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
        asserts on them.
    """

    RADIX = 0  # streaming radix (dim past the DTE slice limit, or no level holds the row) - unsupported
    HEAP = 1  # unsupported
    QUICK = 2  # unsupported
    BITONIC = 3  # unsupported
    KTOP1_AXIS_SPLIT = 4  # k <= 10 - not implemented; routed to the radix kernel
    SCAN = 5  # unsupported
    SORT = 6  # unsupported
    KTOP1_BATCH_SPLIT = 7  # k <= 10 - same as above, routed to the radix kernel
    RADIX_BATCH_SPLIT = 8  # supported: one SIP per row (topsop LEVEL=1)
    KTOP1_SHORTCUT = 9  # k <= 10 - same as above, routed to the radix kernel


_PATTERN_NAMES = {getattr(TopkPattern, name): name for name in dir(TopkPattern) if not name.startswith("_")}


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
    if (level == 1 and n_rows <= NUM_BLOCKS and n_cols >= 256 and _batch_radix_is_support(n_cols, 2)):
        level = 2
    return level


def topk_tuner(n_rows: int, n_cols: int, k: int) -> int:
    if (n_rows >= DTE_SLICE_MAX_INPUT_DIM_SIZE or n_cols >= DTE_SLICE_MAX_INPUT_DIM_SIZE):
        return TopkPattern.RADIX

    if k <= 10:
        if _ktop1_shortcut_is_support(n_rows, n_cols, k):
            return TopkPattern.KTOP1_SHORTCUT
        return TopkPattern.KTOP1_AXIS_SPLIT

    if (_batch_radix_is_support(n_cols, 1) or _batch_radix_is_support(n_cols, 2) or _batch_radix_is_support(n_cols, 3)):
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
    if not (3 <= k <= 10):  # topsop: KTOP1_SHORTCUT_MIN_K .. MAXK
        return False
    if n_cols < 8192:  # topsop: KTOP1_SHORTCUT_MIN_AXIS
        return False
    # topsop: row-bound tables, keyed by axis <= 32768 / axis > 32768.
    row_bound = (48, 48, 48, 88, 88, 96, 96, 96) if n_cols <= 32768 \
        else (20, 24, 32, 48, 48, 48, 48, 96)
    if n_rows <= 0 or n_rows > row_bound[k - 3]:
        return False
    workers = (NUM_BLOCKS * NUM_THREADS * MAX_THREAD_NUM * KTOP1_SHORTCUT_SUBTHREAD_NUM)
    return workers >= n_rows


def _launch_batch_radix(x, y_vals, y_idx, n_rows, n_cols, k):
    """One SIP per row, persistent over rows, no cross-SIP sync.

    Each program owns a strided row set (row_base = program_id,
    row_stride = num_programs) and walks it on the Triton side: S1 stages the
    row with tle.gpu.copy, the raw call runs S1 convert .. S4 in place, S5
    reads the staging back and stores it. One launch with grid =
    min(SIP_COUNT, M) blocks of a single thread (4 subthreads each) covers all
    M rows.

    BLOCK_N = max(128, next_pow2(N)) is the copy tile: a power of two so
    tl.arange can express it, per-N so the staged transfer is the row's own
    size (a fixed 32768 tile would pad 32x for N=1024), and at least one
    128-lane vector chunk because S1 convert rewrites ceil(N/128) whole
    chunks -- the lanes in [N, 128) must arrive as -inf, not as whatever
    VDMEM held. It is a constexpr, so the kernel is specialized per N (same
    trade-off as the S5 tile).
    """
    if n_cols > L1_MAX_COLS:
        raise ValueError(f"N={n_cols} exceeds the one-SIP L1 capacity N<={L1_MAX_COLS}")
    grid_size = min(MAX_GRID, n_rows)
    topk_mradix_l1_launch[(grid_size, )](
        x, y_vals, y_idx, n_rows, n_cols, k, BLOCK_N=max(128, triton.next_power_of_2(n_cols)),
        BLOCK_SIZE=1,  # type: ignore[arg-type]
        num_warps=NUM_WARPS,  # type: ignore[call-arg]
        num_stages=1,  # type: ignore[call-arg]
    )
    return y_vals, y_idx


def _assert_close_vals(a: torch.Tensor, b: torch.Tensor, tag: str):
    """Compare two tiles with torch.testing.assert_close, on the CPU.

    Both operands are moved to the CPU before comparing, because a device-side
    comparison ends up in a torch.all(bool) reduction, and that reduction on
    GCU400 poisons the device: the NEXT cooperative launch dies with "Sip
    exception" (KMD abort).
    """
    if a.shape != b.shape:
        raise AssertionError(f"{tag}: shape mismatch {a.shape} vs {b.shape}")
    torch.testing.assert_close(a.to("cpu"), b.to("cpu"), rtol=0.0, atol=1e-3, equal_nan=True, msg=tag)


def _check_topk(x: torch.Tensor, y_vals: torch.Tensor, y_idx: torch.Tensor, tag: str):
    """Verify one topk result against torch.topk, order-insensitively.

    The returned order is unspecified (reclaimed order), so the values are
    compared as sorted multisets. The index stream is pinned separately: every
    returned index must carry exactly the value returned next to it.
    """
    k = y_vals.shape[1]
    t_vals, _ = torch.topk(x, k, dim=1, sorted=False)
    _assert_close_vals(
        torch.sort(y_vals, dim=1, descending=True).values,
        torch.sort(t_vals, dim=1, descending=True).values, f"values {tag}")
    _assert_close_vals(x.gather(1, y_idx.to(torch.int64)), y_vals, f"indices {tag}")


def run_correctness_on(x: torch.Tensor, k: int, tag: str):
    """Run one topk and verify it against torch.topk."""
    y_vals, y_idx = tle_raw_topk_mradix(x, k)
    _check_topk(x, y_vals, y_idx, tag)
    print(f"MRADIX TopK correctness PASSED ({tag})")


def run_correctness(m: int, n: int, k: int, dtype: torch.dtype = torch.float32):
    """Uniform random floats."""
    torch.manual_seed(42)
    run_correctness_on(torch.rand((m, n), device=DEVICE, dtype=dtype), k, f"M={m} N={n} K={k}")


def run_benchmark(m: int, n: int, k: int, dtype: torch.dtype = torch.float32):
    """Quick manual benchmark (kept for ad-hoc debugging).

    The canonical performance harness is the perf_report benchmark below
    (test_benchmark_topk_*), whose numbers cross-check
    against topsprof --print-gcu-trace kernel times.
    """
    torch.manual_seed(42)
    x = torch.rand((m, n), device=DEVICE, dtype=dtype)

    def fn_mradix():
        return tle_raw_topk_mradix(x, k)

    def fn_torch():
        return torch.topk(x, k, dim=1, sorted=False)

    ms_tle_raw = triton.testing.do_bench(fn_mradix, warmup=50, rep=200)
    ms_torch = triton.testing.do_bench(fn_torch, warmup=50, rep=200)
    # ratio = kernel / torch (<1: this kernel is faster) and speedup = its
    # reciprocal, torch / kernel (>1: this kernel is faster).
    ratio = ms_tle_raw / ms_torch
    print(f"[M={m:4d}, N={n:5d}, K={k:3d}]  "
          f"torch: {ms_torch:.3f} ms  |  "
          f"struct+raw: {ms_tle_raw:.3f} ms  |  "
          f"ratio: {ratio:.2f}x  |  speedup: {1.0 / ratio:.2f}x")


# ==============================================================================
# Performance Tests (perf_report + do_bench(quantiles))
# ==============================================================================
#
# Measurement methodology notes (validated against topsprof --print-gcu-trace
# kernel times, which are the ground truth):
#   - do_bench times HOST-side elapsed time per fn() call = pure kernel time
#     (topsprof) + a fixed launch/host overhead. The pairs quoted below are
#     from the sibling PURE-RAW implementation (04-topk-mradix.py) and are kept
#     only to size that overhead, not as numbers for this kernel:
#     (128,32768,256) 133 us kernel vs 140 us do_bench (+7 us);
#     (8,4096,32) 9.4 us kernel vs 17.0 us do_bench (+7.6 us).
#     TODO(perf-alignment): re-measure this Struct+Raw kernel against those
#     pairs and keep both columns in perf_results/*.csv, so the regression
#     against the pure-raw baseline is visible row by row.
#     kernels >> overhead the two agree, and for tiny kernels do_bench is
#     launch-overhead-dominated - read those rows as end-to-end cost, not
#     kernel cost.
#   - do_bench's built-in n_warmup/n_repeat estimation plus its 256MB L2
#     cache-clear between reps make single-call timing stable (p20/p80
#     spread < 3% here), so no extra inner loop is needed. An inner loop
#     (K>1 calls per timed region) must be avoided: measured K=20 makes
#     torch collapse toward steady-state throughput (28 -> 20 us) while
#     this kernel's per-call staging + launch instead grows
#     (28 -> 68 us) - either way it stops matching topsprof's
#     per-invocation kernel times.
#   - Correctness inside the benchmark goes through _assert_close_vals, which
#     compares on the CPU: a device-side torch.allclose/torch.all bool
#     reduction poisons the device so the NEXT cooperative launch aborts.

# Main benchmark set, grouped by the topsop-aligned TopkPattern each shape
# routes to (verified against topk_tuner) - exactly the 3 supported groups:
#   KTOP1 (k <= 10): KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT / KTOP1_BATCH_SPLIT
#     share one dispatch arm that runs the radix kernel; the label only
#     records which topsop pattern the shape maps to.
#     (1,16384,8) sits inside the ktop1_shortcut admission window (k=8,
#     N=16384, 1 <= row_bound 96) so the SHORTCUT arm is exercised too.
#   RADIX_BATCH_SPLIT (k > 10): all k > 10 shapes that fit one SIP.
#   LARGE_ROW: the wide-row shapes (N=16384), kept as a separate table
#     because they are the closest thing to a bandwidth-bound regime.
BENCH_CASES_KTOP1 = [(1, 1024, 8), (8, 1024, 8), (16, 1024, 8), (64, 128, 8), (1, 16384, 8),  # KTOP1_SHORTCUT window
                     ]
BENCH_CASES_L1 = [
    # N=4096: small-batch controls (M below SIP_COUNT, so a single wave).
    (1, 4096, 32),
    (1, 4096, 64),
    (4, 4096, 32),
    (4, 4096, 64),
    (8, 4096, 32),
    (8, 4096, 64),
    # Larger M and the widest supported row (N=32768, K=256).
    (64, 1024, 32),
    (64, 8192, 128),
    (128, 32768, 256),
]
# Wide-row group: one SIP still owns a whole row, so these are the shapes
# where the kernel is closest to VDMEM-bandwidth bound.
BENCH_CASES_LARGE_ROW = [
    (1, 16384, 32),
    (1, 16384, 64),
    (4, 16384, 32),
    (4, 16384, 64),
    (8, 16384, 32),
    (8, 16384, 64),
]
# Unaligned-K group: same 8 shapes as the correctness group of that name, so a
# regression shows up in both. Row strides are 400B (K=100), 516B (129),
# 544B (136), 800B (200), 928B (232), 1200B (300), 1552B (388) and 2080B
# (520); K=100 stays on the K < VEC_LEN deslice writeback. This is the group
# that stresses S5 (row writeback) hardest.
BENCH_CASES_UNALIGNED_K = [
    (32, 8192, 100),
    (32, 8192, 129),
    (32, 8192, 136),
    (32, 8192, 200),
    (16, 4096, 232),
    (32, 8192, 300),
    (16, 8192, 388),
    (32, 8192, 520),
]


def _bench_one(provider, m, n, k):
    """Benchmark one (M, N, K) x provider; returns (median, p20, p80) in ms.

    Input tensor is allocated fresh per call (matmul-style) so the result
    does not depend on tensor reuse across x_vals.
    """
    torch.manual_seed(42)
    x = torch.rand((m, n), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        return triton.testing.do_bench(lambda: torch.topk(x, k, dim=1, sorted=False), quantiles=quantiles)

    # Any other provider label runs this kernel: verify correctness once per
    # (shape, provider) before timing (host-side assert_close, see
    # _assert_close_vals).
    y_vals, y_idx = tle_raw_topk_mradix(x, k)
    _check_topk(x, y_vals, y_idx, f"bench M={m} N={n} K={k}")

    return triton.testing.do_bench(lambda: tle_raw_topk_mradix(x, k), quantiles=quantiles)


def _make_topk_benchmark(plot_name, cases):
    return triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=list(cases),
            line_arg='provider',
            line_vals=['torch', 'struct-raw'],
            line_names=["Torch", "Struct+Raw"],
            styles=[('green', '-'), ('blue', '-')],
            ylabel="us",
            plot_name=plot_name,
            args={},
        ))(lambda M, N, K, provider: tuple(us * 1000.0 for us in _bench_one(provider, M, N, K)))


class TestTopkPerf:
    """Performance benchmark for the Struct+Raw MRADIX topk (topsprof-aligned)."""

    @pytest.fixture(autouse=True)
    def setup_benchmark(self, tmp_path='./perf_results'):
        # matplotlib saves the PNG before benchmark.run() creates save_path,
        # so the directory must already exist here.
        os.makedirs(str(tmp_path), exist_ok=True)
        self._save_path = str(tmp_path)

    def test_benchmark_topk_ktop1(self):
        """KTOP1 group (k <= 10; KTOP1_SHORTCUT / KTOP1_AXIS_SPLIT /
        KTOP1_BATCH_SPLIT all run the L1 radix kernel, see dispatch)."""
        benchmark = _make_topk_benchmark("gcu400-struct-raw-topk-mradix-ktop1", BENCH_CASES_KTOP1)
        benchmark.run(show_plots=True, print_data=True, save_path=self._save_path)

    def test_benchmark_topk_l1(self):
        """RADIX_BATCH_SPLIT L1 group."""
        benchmark = _make_topk_benchmark("gcu400-struct-raw-topk-mradix-l1", BENCH_CASES_L1)
        benchmark.run(show_plots=True, print_data=True, save_path=self._save_path)

    def test_benchmark_topk_large_row(self):
        """Wide-row (N=16384) group."""
        benchmark = _make_topk_benchmark("gcu400-struct-raw-topk-mradix-large-row", BENCH_CASES_LARGE_ROW)
        benchmark.run(show_plots=True, print_data=True, save_path=self._save_path)

    def test_benchmark_topk_unaligned_k(self):
        """Unaligned-K group (K % 128 != 0, misaligned output rows)."""
        benchmark = _make_topk_benchmark("gcu400-struct-raw-topk-mradix-unaligned-k", BENCH_CASES_UNALIGNED_K)
        benchmark.run(show_plots=True, print_data=True, save_path=self._save_path)


def run_correctness_mixed(m: int, n: int, k: int):
    """Mixed positive/negative values (randn)."""
    torch.manual_seed(123)
    run_correctness_on(torch.randn((m, n), device=DEVICE, dtype=torch.float32), k, f"M={m} N={n} K={k} mixed +/-")


def run_correctness_ties(m: int, n: int, k: int):
    """Duplicate values, i.e. ties at the k-th value (the split path).

    Few distinct levels make n_ge > K on most rows; the all-equal tensor is the
    extreme (n_ge == N, n_gt == 0). Which duplicate ends up in the output is
    unspecified, so _check_topk compares value multisets.
    """
    torch.manual_seed(7)
    run_correctness_on(torch.randint(0, 4, (m, n)).to(torch.float32).to(DEVICE), k, f"M={m} N={n} K={k} ties/4 levels")
    run_correctness_on(torch.full((m, n), 0.25, dtype=torch.float32, device=DEVICE), k,
                       f"M={m} N={n} K={k} ties/all-equal")


if __name__ == "__main__":
    if "--perf" in sys.argv:
        # perf_report tables + CSV/PNG under ./perf_results/, cross-checkable with topsprof.
        pytest.main([__file__, "-v", "-k", "benchmark", "--noconftest"])
        sys.exit(0)

    print(f"SIP_COUNT={SIP_COUNT}, NUM_WARPS={NUM_WARPS}, MAX_GRID={MAX_GRID}")
    print()

    # Correctness sets, grouped by the TopkPattern topk_tuner routes each
    # shape to - the same 3 groups as the benchmark lists above. All KTOP1
    # shapes (k <= 10) run the L1 radix kernel (RADIX_BATCH_SPLIT measured
    # better, see dispatch), so the L1 radix path is exercised by both the
    # KTOP1 and the RADIX_BATCH_SPLIT groups.
    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (KTOP1 group, k<=10)")
    print("=" * 60)
    for m, n, k in [(1, 8, 3), (2, 16, 4), (4, 128, 8), (8, 256, 4), (8, 1024, 8), (1, 4096, 10), (16, 256, 4),
                    (32, 1024, 8), (64, 128, 8), (128, 256, 4), (1, 16384, 8)]:  # KTOP1_SHORTCUT window
        run_correctness(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (RADIX_BATCH_SPLIT group, k>10)")
    print("=" * 60)
    for m, n, k in [(1, 4096, 32), (1, 4096, 64), (4, 4096, 32), (8, 4096, 64), (2, 8192, 16), (8, 8192, 128),
                    (64, 8192, 128), (128, 32768, 256)]:
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
    for m, n, k in [(32, 8192, 100), (32, 8192, 129), (32, 8192, 136), (32, 8192, 200), (16, 4096, 232),
                    (32, 8192, 300), (16, 8192, 388), (32, 8192, 520)]:
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
    for m, n, k in [(1, 256, 8), (8, 1024, 32), (4, 4096, 128), (2, 8192, 512)]:
        run_correctness_ties(m, n, k)
    print()

    print("=" * 60)
    print("MRADIX TopK - Correctness Tests (large-row group)")
    print("=" * 60)
    for m, n, k in [(1, 16384, 32), (1, 16384, 64), (4, 16384, 32), (4, 16384, 64), (8, 16384, 32), (8, 16384, 64)]:
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
