# dsa_kpool_topk_transform — TLE implementation (v2: smem histogram + atomics)
#
# 算子名: dsa_kpool_topk_transform
# 内核名: kpool_topk_transform_kernel
# 接口名: fast_kpool_topk_transform_fused(score, lengths, output, pool_size,
#         page_table=None, topk_indices_offset=None, row_starts=None,
#         seq_lens=None, page_table_row_index=None) -> None
# 对比基线: mcoplib 原生 torch.ops.sgl_kernel.kpool_topk_transform
#          （op/sglang/jit_kernels/kpool_topk_transform.cu）
#
# 为什么不用 tl.histogram / tl.cumsum（prefill_63ki, B=1024 x W=16126 实测）:
#   tl.histogram +142 us/遍   （按 bin 的归约循环，成本随 bin 数线性增长）
#   全行 [4096] tl.cumsum  +123 us/遍   （Hillis-Steele，用作位置前缀和太贵）
#   共享内存 relaxed cta-scope 原子直方图  +19 us/遍   <- 采用
#   全局指针 atomic 取位置                 +165 us/遍
#   单个 block 归约 / [2048] tl.cumsum     ~3.9 / ~18.8 us  <- 小规模时才划算
# 依据: FlagGems src/flag_gems/fused/DSA/bin_topk.py 的 TLE 实现同样用
#   tle.gpu.alloc(smem) + tl.atomic_add(sem="relaxed", scope="cta") 建直方图。
#
# v2 算法（每行一个 CTA，两遍全局读，与 MACA 原生一致）:
#   遍 1: ordered-fp32 key 的高 11 位（每个 binade 4 个桶）在 smem 里建直方图。
#   阈值: 一次 tle.cumsum 扫 2048 个 bin -> 阈值桶 t_bin / count_above /
#         remain / n_cand。
#   遍 2: d > t_bin 的元素把行内下标压紧存进 smem（槽号即输出位置）；
#         d == t_bin 的候选（n_cand <= CAP）同样压紧存进 smem。
#         两轮都用 smem 原子取槽号，遍 2 不写全局。
#   发射: 已选集合按 [CAPP] 稠密块发射（位置 0..count_above），
#         候选按 [CAP] 稠密块接着发射。稠密块比 [BLOCK] 宽散射写快约 4 倍。
#   refine: 在 [CAP] 寄存器块上做 2 轮 11-bit MSD radix（smem 直方图 +
#         [2048] tl.cumsum 选阈值），得到精确 cutoff key / 剩余并列数。
#   发射候选: 原子位置接着 count_above 往下写，填满 [0, group_topk)。
#   兜底: n_cand > CAP 时走 v1 已验证的 4 遍全行 rescan（精确，罕见）。
#
# 语义与 mcoplib 单元测试 test_kpool_topk_transform.py 的
# reference_kpool_topk_transform 对齐；选中的组之间无序（原子位置天然无序）。

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton import knobs


# ---------------------------------------------------------------------------
# key transforms
# ---------------------------------------------------------------------------


@triton.jit
def _okey(x):
    """Ordered uint32 key: unsigned order == float order (non-NaN)."""
    u = x.to(tl.uint32, bitcast=True)
    sign = u >> 31
    flip = (0 - sign) | 0x80000000
    return u ^ flip


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@triton.jit
def _emit_one(row_out, pos, idx, valid, off, table_base,
              HAS_PT: tl.constexpr, HAS_OFF: tl.constexpr,
              POOL: tl.constexpr):
    """Write one POOL-wide group per selected element, one column at a time."""
    for q in tl.static_range(POOL):
        raw = idx * POOL + q
        if HAS_PT:
            val = tl.load(table_base + raw, mask=valid, other=-1)
        elif HAS_OFF:
            val = raw + off
        else:
            val = raw
        tl.store(row_out + pos * POOL + q, val, mask=valid)


@triton.jit
def _emit_tile(row_out, pos, idx, valid, off, table_base,
               HAS_PT: tl.constexpr, HAS_OFF: tl.constexpr,
               POOL: tl.constexpr, POOL_PAD: tl.constexpr):
    """Same output as `_emit_one`, as one 2-D block load + one block store.

    With the per-column loop the compiler has to keep each load ordered before
    its store (it cannot prove `table_base` and `row_out` do not alias), so the
    POOL global round trips serialize: measured 1.65 us for the selected-set
    emit and 2.86 us for the candidate emit on decode_3ki, i.e. ~600 cycles per
    column -- one dependent miss each.  As a [N, POOL_PAD] tile the loads all
    issue at once and the stores follow.
    """
    qs = tl.arange(0, POOL_PAD)
    raws = idx[:, None] * POOL + qs[None, :]
    keep = valid[:, None] & (qs[None, :] < POOL)
    if HAS_PT:
        val = tl.load(table_base + raws, mask=keep, other=-1)
    elif HAS_OFF:
        val = raws + off
    else:
        val = raws
    tl.store(row_out + pos[:, None] * POOL + qs[None, :], val, mask=keep)


@triton.jit
def _emit(row_out, pos, idx, valid, off, table_base,
          HAS_PT: tl.constexpr, HAS_OFF: tl.constexpr,
          POOL: tl.constexpr, POOL_PAD: tl.constexpr, TILE: tl.constexpr):
    """Column-loop or 2-D tile emit.

    The tile form is only a win where the emit is latency-bound on the page
    table (the 2-row decode shape, -1.7 us); on the 1024-lane shapes it costs
    2-12% because the [CAP, POOL_PAD] block needs its own staging and the
    emit there is not latency-bound.
    """
    if TILE:
        _emit_tile(row_out, pos, idx, valid, off, table_base, HAS_PT, HAS_OFF,
                   POOL, POOL_PAD)
    else:
        _emit_one(row_out, pos, idx, valid, off, table_base, HAS_PT, HAS_OFF,
                  POOL)


@triton.jit
def _scan_select(cnt, remain, NB: tl.constexpr, LOGB: tl.constexpr):
    """Largest bin b with count(d >= b) >= remain, plus count(d > b).

    Replaces a prefix scan: on this backend `tl.cumsum` over a [256] block read
    from shared memory fails to lower, and scans over big blocks are slow
    (~30x a hand scan).  LOGB masked sums cost far less than NB-lane scans.
    """
    bk = tl.arange(0, NB)
    lo = 0
    hi = NB - 1
    for _ in tl.static_range(LOGB):
        mid = (lo + hi + 1) // 2
        ok = tl.sum(tl.where(bk >= mid, cnt, 0)) >= remain
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid - 1)
    n_gt = tl.sum(tl.where(bk > lo, cnt, 0))
    return lo, n_gt


@triton.jit
def _rescan_refine(row_score, length, n_tiles, t_bin, remain0,
                   SHIFT: tl.constexpr, BLOCK: tl.constexpr,
                   BINS: tl.constexpr):
    """Overflow fallback (v1, exact): 4 full-row rescan rounds."""
    pref = 0
    count_gt = 0
    remain = remain0
    for l in tl.static_range(4):
        shift = 24 - 8 * l
        hist_k = tl.zeros([256], tl.int32)
        n_active = 0
        for t in range(0, n_tiles):
            idx = t * BLOCK + tl.arange(0, BLOCK)
            m = idx < length
            x = tl.load(row_score + idx, mask=m, other=float("-inf"))
            key = _okey(x)
            # mask with the *configured* histogram width: hard-coding 0x7FF
            # here made `b == t_bin` unreachable when the coarse pass ran with
            # bins < 2048, which then emitted past the row (illegal access).
            b = (key >> SHIFT) & (BINS - 1)
            if l == 0:
                active = (b == t_bin) & m
            else:
                active = (b == t_bin) & (
                    (key >> (shift + 8)) == (pref >> (shift + 8))) & m
            byte_v = tl.where(active, (key >> shift) & 0xFF, 0)
            hist_k += tl.histogram(byte_v.to(tl.int32), 256)
            n_active += tl.sum(active.to(tl.int32))
        n_inactive = n_tiles * BLOCK - n_active
        hist_k = hist_k - tl.where(tl.arange(0, 256) < 1, n_inactive, 0)
        tb, n_gt = _scan_select(hist_k, remain, 256, 8)
        count_gt += n_gt
        remain -= n_gt
        # uint32 on purpose: `pref >> shift` must be a logical shift.
        pref = pref | (tb.to(tl.uint32) << shift)
    return pref, count_gt, remain


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------


@triton.jit
def kpool_topk_transform_kernel(
    score_ptr,
    lengths_ptr,
    out_ptr,
    page_table_ptr,
    offset_ptr,
    starts_ptr,
    seq_ptr,
    trow_ptr,
    width,
    out_cols,
    table_width,
    token_topk,
    group_topk,
    POOL: tl.constexpr,
    POOL_PAD: tl.constexpr,
    HAS_PT: tl.constexpr,
    HAS_OFF: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    HAS_TROW: tl.constexpr,
    BINS: tl.constexpr,
    LOG2_BINS: tl.constexpr,
    SHIFT0: tl.constexpr,
    HISTN: tl.constexpr,
    R0S: tl.constexpr,
    R0N: tl.constexpr,
    R1N: tl.constexpr,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    CAPP: tl.constexpr,
    WIDE: tl.constexpr,
    DIG: tl.constexpr,
    DIGW: tl.constexpr,
    SAMP: tl.constexpr,
    DROP1: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row)
    start = tl.load(starts_ptr + row) if HAS_STARTS else 0
    off = tl.load(offset_ptr + row) if HAS_OFF else 0
    tail_n = tl.load(seq_ptr + row) % POOL if HAS_TAIL else 0

    row_score = score_ptr + row.to(tl.int64) * width + start.to(tl.int64)
    row_out = out_ptr + row.to(tl.int64) * out_cols
    table_base = (
        page_table_ptr
        + (tl.load(trow_ptr + row) if HAS_TROW else row).to(tl.int64) * table_width
    )

    lane = tl.arange(0, BLOCK)
    ones = tl.full([BLOCK], 1, tl.int32)

    # ---------------- early path: length <= group_topk (identity) -----------
    if length <= group_topk:
        limit = length * POOL + tail_n
        for c0 in range(0, out_cols, BLOCK):
            col = c0 + lane
            m = col < out_cols
            live = col < limit
            if HAS_PT:
                val = tl.load(table_base + col, mask=live, other=-1)
            elif HAS_OFF:
                val = col + off
            else:
                val = col
            val = tl.where(live, val, -1)
            tl.store(row_out + col, val, mask=m)
        return

    # ---------------- drop-one path: length == group_topk + 1 ---------------
    # Then the selected set is "every group except the single minimum", so the
    # whole histogram + radix refine pipeline is unnecessary.  Compiled only
    # when the row is narrow enough for it to be reachable (`DROP1`, decided by
    # the wrapper) and when there is no tail region to append -- the tail writer
    # sits after the general path, so a tailed row must not return here.  A
    # tied minimum falls through to the exact general path: which tied index
    # the reference drops is not defined by the contract.
    if DROP1 == 1 and length == group_topk + 1:
        d1l = tl.arange(0, 2 * CAPP)
        d1m = d1l < length
        d1x = tl.load(row_score + d1l, mask=d1m, other=float("inf"))
        d1min, d1idx = tl.min(d1x, axis=0, return_indices=True)
        d1cnt = tl.sum((d1m & (d1x == d1min)).to(tl.int32))
        if d1cnt == 1:
            # dense output slot = lane index with the removed one skipped
            d1valid = d1m & (d1l != d1idx)
            d1slot = d1l - (d1l > d1idx).to(tl.int32)
            _emit_one(row_out, d1slot, d1l, d1valid, off, table_base,
                      HAS_PT, HAS_OFF, POOL)
            return

    # ---------------- shared memory -----------------------------------------
    s_hist = tle.gpu.alloc([HISTN], dtype=tl.int32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    hist_ptr = tle.gpu.local_ptr(s_hist, (0,))
    # [0] output position counter, [1] staging counter, [2] tie counter
    s_ctr = tle.gpu.alloc([4], dtype=tl.int32, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ctr_ptr = tle.gpu.local_ptr(s_ctr, (0,))
    # Two staging lists, one store. The selected set (d > t_bin) lands in
    # s_sel, where the slot IS the output position (CAPP = next_pow2(group_topk)
    # bounds it, since count_above <= group_topk), so the emit is dense and
    # needs no position atomic. The threshold bin (d == t_bin) lands in
    # s_sidx, which is what the refine reads.
    #
    # r23: the atomic address already picks the counter (`1 + eq`), so it also
    # picks the list via `1 + 2 * (pos + ...)`-style pointer select below, and
    # the lane issues ONE predicated store. The r22 shape issued three
    # (sel/sidx/skey) and those extra issued vector-memory instructions -- the
    # C500 grants one vector-memory dispatch slot per cycle per PEU, and a
    # predicated-off lane still occupies its slot -- were 46 % of the whole
    # kernel at 32768-wide rows.
    # Both lists hold element indices, and an index is < width <= 65536: two
    # bytes, not four. The width matters because shared memory is the
    # occupancy knob -- at 16 bits the two lists cost CAPP + CAP words
    # instead of their double, which is what lets a 2048-wide tile fit three
    # CTAs per AP. Counters stay 32-bit; only the staged payload narrows.
    s_sel = tle.gpu.alloc([CAPP], dtype=tl.uint16, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    sel_ptr = tle.gpu.local_ptr(s_sel, (0,))
    s_sidx = tle.gpu.alloc([CAP], dtype=tl.uint16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    sidx_ptr = tle.gpu.local_ptr(s_sidx, (0,))
    # 11-bit coarse digit of every row element, written by the histogram pass
    # and read back by the classification pass, so the row is read from global
    # exactly once. Two bytes per element: the digit is 0..BINS-1. Declared
    # last, and only when the wrapper could fit it (DIG): allocating it
    # unconditionally costs ~6% on the 1024-lane shapes even though nothing
    # there touches it.
    if DIG:
        s_dig = tle.gpu.alloc([DIGW], dtype=tl.uint16, layout=None,
                              scope=tle.gpu.smem, nv_mma_shared_layout=False)
        dig_ptr = tle.gpu.local_ptr(s_dig, (0,))

    n_tiles = tl.cdiv(length, BLOCK)
    hbins = tl.arange(0, BINS)

    # ---------------- pass 1: 11-bit ordered-key histogram -------------------
    for c in tl.range(0, tl.cdiv(BINS, BLOCK)):
        bins_off = c * BLOCK + lane
        tl.store(hist_ptr + bins_off, 0, mask=bins_off < BINS)
    cslots = tl.arange(0, 4)
    tl.store(ctr_ptr + cslots, 0)
    tl.debug_barrier()

    if WIDE > 0:
        # Whole row in one block: the ordered key stays in registers and the
        # staging pass below reuses it, so the row is read once instead of
        # twice (~49 us per read on prefill_63ki, and the second read does
        # not hit L2 -- the in-flight working set exceeds the 8 MB L2).
        lane_w = tl.arange(0, WIDE)
        ones_w = tl.full([WIDE], 1, tl.int32)
        mw = lane_w < length
        key_w = _okey(tl.load(row_score + lane_w, mask=mw, other=0.0))
        d_w = (tl.where(mw, key_w, 0) >> SHIFT0) & (BINS - 1)
        tl.atomic_add(hist_ptr + d_w, ones_w, mask=mw, sem="relaxed",
                      scope="cta")
    else:
        if SAMP > 0:
            # ---------------- sampled single-read path ----------------------
            # A 32768-wide row is 128 KB and one pass over it is ~45 us of the
            # ~125 us the whole kernel takes at B=256. The second pass exists
            # only because the threshold bin is not known until the histogram
            # is complete. Sample `SAMP` tiles first, take the threshold from
            # them, and then make the complete pass do the histogram AND the
            # classification together: the row is read once instead of twice.
            #
            # The sampled threshold is only an estimate, so it is never trusted
            # for anything but *staging*: the complete pass still builds the
            # exact histogram, and the fast path is taken only when the exact
            # counts it yields prove the staged set contains the answer (see
            # the guard below). A wrong estimate costs the exact rescan, not
            # correctness.
            for t in tl.range(0, SAMP):
                # sample the row's tail; the complete pass walks backwards
                # from the same end, so those tiles are still in cache
                st = (n_tiles - 1) - (t * (n_tiles - 1)) // SAMP
                idx = st * BLOCK + lane
                m = idx < length
                x = tl.load(row_score + idx, mask=m, other=0.0)
                d = (tl.where(m, _okey(x), 0) >> SHIFT0) & (BINS - 1)
                tl.atomic_add(hist_ptr + d, ones, mask=m, sem="relaxed",
                              scope="cta")
            tl.debug_barrier()
            if HISTN == BINS:
                hs = tle.gpu.metax.local_load(s_hist)
            else:
                hs = tl.atomic_add(hist_ptr + hbins,
                                   tl.zeros([BINS], tl.int32),
                                   sem="relaxed", scope="cta")
            pres = tl.cumsum(hs, axis=0) - hs
            # Aim the sample at the group_topk-th element itself, i.e. the
            # cutoff the exact histogram would pick, scaled to the sample.
            # Aiming past it is *not* the safe direction: the estimate is a
            # digit, and a digit holds a good fraction of the row, so asking
            # for 1.5x put count_above at 723 of the 512 allowed on the normal
            # scores the bench uses (82/256 rows then missed the guard, and a
            # 1.0x aim passes 256/256 with count_above 162-236 and n_cand
            # 490-612). Aiming short fails the other way, with a staged set
            # too small to fill group_topk.
            tgt = (group_topk * (SAMP * BLOCK)) // length
            t_bin = tl.max(tl.where(pres <= SAMP * BLOCK - tgt, hbins, -1))
            # No clear here: the merged pass below does not touch the histogram
            # any more. The sample selected the only thing the histogram was
            # needed for (`t_bin`), and the exact populations are counted in
            # registers instead -- the merged pass used to build the whole
            # 2048-bin histogram with a scattered shared-memory atomic per
            # element, which is what made the single-read version only 12 us
            # cheaper than the two-read one at B=256 while it saved a full read.
            # The overflow fallback rebuilds the histogram from scratch if the
            # estimate turns out to be unusable.
            acc_a = tl.zeros([BLOCK], tl.int32)
            acc_e = tl.zeros([BLOCK], tl.int32)
            for t in tl.range(0, n_tiles):
                idx = (n_tiles - 1 - t) * BLOCK + lane
                m = idx < length
                x = tl.load(row_score + idx, mask=m, other=0.0)
                key = _okey(x)
                d = (tl.where(m, key, 0) >> SHIFT0) & (BINS - 1)
                eq = m & (d == t_bin)
                above = m & (d > t_bin)
                keep = above | eq
                acc_a += above.to(tl.int32)
                acc_e += eq.to(tl.int32)
                pos = tl.atomic_add(ctr_ptr + 1 + eq.to(tl.int32), ones,
                                    mask=keep, sem="relaxed", scope="cta")
                # the threshold here is an estimate, so a slot can run past the end
                # of its list before the guard below rejects the row: saturate instead
                # of writing past the array (the row is rescanned, so a saturated
                # slot is never read).
                pos = tl.minimum(pos, tl.where(eq, CAP - 1, CAPP - 1))
                tl.store(tl.where(eq, sidx_ptr + pos, sel_ptr + pos),
                         idx.to(tl.uint16), mask=keep)
            # The two populations the guard and the emit need, as two block
            # reductions over per-lane accumulators (one add per element in the
            # loop). Packing them into one reduction is not safe here: the
            # packed sum overflows int32 once the tie bin holds 32768 elements,
            # and those are exactly the rows that must fall back.
            count_above = tl.sum(acc_a)
            n_cand = tl.sum(acc_e)
        else:
            for t in tl.range(0, n_tiles):
                # Walk the tiles backwards here, forwards in pass 2: the tiles
                # pass 1 touches last are then the ones pass 2 asks for first
                # and are still in L2 (the in-flight set is ~128 KB per row,
                # far more than the 8 MB L2 divided by the ~200 resident CTAs,
                # so only the most recent tiles can hit). The histogram is a
                # sum, so the order cannot change its result, and pass 2 keeps
                # its order, so nothing observable moves. Measured: -1.7 us at
                # B=256, -3.8 us at B=1024 on the 32768-wide rows.
                idx = (n_tiles - 1 - t) * BLOCK + lane
                m = idx < length
                x = tl.load(row_score + idx, mask=m, other=0.0)
                d = (tl.where(m, _okey(x), 0) >> SHIFT0) & (BINS - 1)
                tl.atomic_add(hist_ptr + d, ones, mask=m, sem="relaxed",
                              scope="cta")
                if DIG:
                    # keep the digit so pass 2 needs no second global read.
                    # The array is a power-of-two window (the backend requires
                    # it), so a row longer than the window keeps the digit for
                    # the tiles that fit and pass 2 re-reads the rest; the
                    # branch is uniform per tile.
                    tl.store(dig_ptr + idx, d.to(tl.uint16), mask=m)
    tl.debug_barrier()

    if WIDE > 0 or SAMP == 0:
        # ------------- threshold from the histogram (not on the sampled path) -
        # Read the bins back with an atomic add of zero, not tl.load: this
        # backend's optimizer does not model a relaxed smem atomic RMW as a
        # write, so a plain (even volatile) load is hoisted above the
        # accumulation loop and returns the pre-clear zeros -- observed as an
        # all-zero histogram.
        # tle.gpu.metax.local_load reads the same buffer the atomics wrote, so
        # the optimizer cannot hoist it above them (the plain tl.load bug) and
        # it costs ~0.25 us less than the atomic read-back.
        if HISTN == BINS:
            hcnt = tle.gpu.metax.local_load(s_hist)
        else:
            hcnt = tl.atomic_add(hist_ptr + hbins, tl.zeros([BINS], tl.int32),
                                 sem="relaxed", scope="cta")
        # Threshold bin: the largest b with count(d >= b) >= group_topk, i.e.
        # the largest b whose exclusive prefix stays under `length - group_topk`.
        # Every element is histogrammed exactly once, so the bin total is
        # `length` and no reduction is needed to get it. One cumsum (~19 us)
        # beats the 11 masked block sums `_scan_select` would spend per row
        # (~43 us).
        pre = tl.cumsum(hcnt, axis=0) - hcnt
        t_bin = tl.max(tl.where(pre <= length - group_topk, hbins, -1))
        # Two population counts. Each `tl.sum` over [BINS] is a full 512-lane
        # block reduction (~0.8 us, mostly the shared-memory tree, not the 4
        # elements per thread), so where the row block is already in registers
        # they are counted there instead and the two counts ride one reduction
        # as 16-bit halves (length <= 16384, so neither field can carry).
        if WIDE > 0:
            pack = tl.sum(tl.where(mw, (d_w > t_bin).to(tl.int32)
                                   | ((d_w == t_bin).to(tl.int32) << 16), 0))
        else:
            # One packed reduction here too: a [BINS] block sum is dominated by
            # the 512-lane tree, so two of them cost twice one (r14 did this
            # for the register path only).
            #
            # The high field must be saturated before the shift: a tie bin
            # holding >= 2**15 elements would shift into the sign bit and read
            # back as a small (or negative) `n_cand`, which turns the
            # `n_cand <= CAP` guard into a false pass and lets pass 2 stage
            # past s_sidx.  Saturation keeps the guard honest (0x7FFF > CAP)
            # and leaves the low field exact (`count_above < group_topk <=
            # 2**16` for the <= 65536-wide rows this kernel supports).
            pack = tl.sum(tl.where(hbins > t_bin, hcnt, 0)
                          | (tl.minimum(tl.where(hbins == t_bin, hcnt, 0),
                                        0x7FFF) << 16))
        n_cand = pack >> 16
        count_above = pack & 0xFFFF
    # (the sampled path counted both populations in registers during the
    # merged pass; `t_bin`/`n_cand`/`count_above` come from there)
    remain0 = group_topk - count_above

    # The register-resident path is only valid while the row fits the block;
    # a longer row (which the interface implies cannot happen, since a row is
    # `width` columns wide and lengths index into it) falls into the exact
    # rescan fallback instead of silently truncating.
    #
    # The last two terms matter only on the sampled path, where `t_bin` is an
    # estimate. They are exactly the two conditions that make the staged set
    # contain the answer no matter what the estimate was: `count_above` <=
    # group_topk keeps the selected set inside the row, and count_above +
    # n_cand >= group_topk means the staged set can fill it. When both hold,
    # "everything above the bin, plus the top `remain0` of the bin" IS the
    # exact top-k for whatever bin was chosen -- the estimate does not have to
    # be the canonical threshold, it only has to bracket the answer.
    if (n_cand <= CAP) & (length <= WIDE) if WIDE > 0 else (
            (n_cand <= CAP) & (count_above <= group_topk)
            & (count_above + n_cand >= group_topk)):
        # ---------------- pass 2: stage above + stage the threshold bin -----
        # Nothing is written to global here: a [BLOCK]-wide scattered emit with
        # a handful of active lanes costs ~4x what the same stores cost from a
        # dense [CAPP] block (measured: +107 us vs ~17 us on prefill_63ki).
        if SAMP > 0:
            # Already staged: the merged pass above classified the row while
            # it built the histogram, so this pass has nothing left to do.
            pass
        elif WIDE > 0:
            # reuse the register-resident key: no second global read
            # One atomic and one store for both lists: the counter address
            # itself picks the list (ctr[1] for d > t_bin, ctr[2] for the
            # threshold bin) and the slot picks the region, so every element
            # gets a dense slot in its own list at the cost of one RMW and one
            # predicated store.
            d_w = (tl.where(mw, key_w, 0) >> SHIFT0) & (BINS - 1)
            eq_w = mw & (d_w == t_bin)
            keep_w = mw & (d_w >= t_bin)
            pos_w = tl.atomic_add(ctr_ptr + 1 + eq_w.to(tl.int32), ones_w,
                                  mask=keep_w, sem="relaxed", scope="cta")
            tl.store(tl.where(eq_w, sidx_ptr + pos_w, sel_ptr + pos_w),
                     lane_w.to(tl.uint16), mask=keep_w)
        elif DIG:
            # The digit staged by pass 1 classifies the element exactly, so
            # this pass reads 2 bytes per element from shared memory instead of
            # re-reading and re-keying the row from global (~49 us, and the
            # second read does not hit L2 -- the in-flight working set exceeds
            # the 8 MB L2). Only the <= CAP candidates need a full key, and
            # those are gathered by index after the barrier.
            for t in tl.range(0, n_tiles):
                idx = t * BLOCK + lane
                m = idx < length
                d = tl.load(dig_ptr + idx, mask=m, other=0).to(tl.int32)
                eq = m & (d == t_bin)
                keep = m & (d >= t_bin)
                pos = tl.atomic_add(ctr_ptr + 1 + eq.to(tl.int32), ones,
                                    mask=keep, sem="relaxed", scope="cta")
                tl.store(tl.where(eq, sidx_ptr + pos, sel_ptr + pos),
                         idx.to(tl.uint16), mask=keep)
        else:
            # No digits: re-read the row. Same one-atomic/one-store staging as
            # the digit path -- the previous shape of this loop issued three
            # predicated stores per lane (sel + sidx + skey) and that was 46 %
            # of the whole kernel at 32768-wide rows.
            for t in tl.range(0, n_tiles):
                idx = t * BLOCK + lane
                m = idx < length
                x = tl.load(row_score + idx, mask=m, other=0.0)
                key = _okey(x)
                d = (tl.where(m, key, 0) >> SHIFT0) & (BINS - 1)
                eq = m & (d == t_bin)
                keep = m & (d >= t_bin)
                pos = tl.atomic_add(ctr_ptr + 1 + eq.to(tl.int32), ones,
                                    mask=keep, sem="relaxed", scope="cta")
                tl.store(tl.where(eq, sidx_ptr + pos, sel_ptr + pos),
                         idx.to(tl.uint16), mask=keep)
        tl.debug_barrier()

        # ---------------- emit the selected set (dense positions) -----------
        # count_above is the exact population of d > t_bin, i.e. exactly the
        # number of slots pass 2 filled, so slots [0, count_above) are all
        # written and every output position is used once.
        plane = tl.arange(0, CAPP)
        live_ab = plane < count_above
        idx_a = tl.load(sel_ptr + plane, mask=live_ab, other=0).to(tl.int32)
        _emit(row_out, plane, idx_a, live_ab, off, table_base,
              HAS_PT, HAS_OFF, POOL, POOL_PAD, DIG)

        # ---------------- exact radix refine over the staged candidates -----
        clane = tl.arange(0, CAP)
        live = clane < n_cand
        cones = tl.full([CAP], 1, tl.int32)
        idx_b = tl.load(sidx_ptr + clane, mask=live, other=0).to(tl.int32)
        # The candidate keys are gathered by index rather than staged during
        # pass 2: staging them cost a second predicated store over every lane
        # of the row, while the gather touches only the <= CAP candidates.
        key_b = _okey(tl.load(row_score + idx_b, mask=live, other=0.0))

        # The two rounds split the low 21 bits as 10 + 11 (round 0 takes bits
        # 20:11, round 1 bits 10:0), which still covers every bit exactly but
        # keeps round 0 -- the round that always runs -- at 1024 bins: a block
        # scan is priced by width on this backend (D11: 6.0 units at [2048],
        # 2.9 at [1024]). A narrower block through a shared memory pointer
        # does not lower, so 1024 is the smallest usable width.
        r0l = tl.arange(0, R0N)
        # seed the prefix with the shared high bits: round 1 compares
        # `key >> 11` against `pref >> 11`, so without them the active set
        # would be empty and the cutoff would degenerate to a prefix.
        pref = t_bin.to(tl.uint32) << SHIFT0
        remain = remain0
        # ---- round 0: bits 20:11 over every staged candidate ---------------
        # The select is one `tl.cumsum` over the bins rather than LOG2_BINS
        # masked block sums: a block reduction costs ~3.9 us here, so 11 sums
        # per round were the single most expensive thing in the kernel. Every
        # candidate is staged, so the bin total is n_cand and needs no
        # reduction.
        d0 = (key_b >> R0S) & (R0N - 1)
        tl.store(hist_ptr + r0l, 0)
        tl.debug_barrier()
        tl.atomic_add(hist_ptr + d0, cones, mask=live,
                      sem="relaxed", scope="cta")
        tl.debug_barrier()
        cnt0 = tl.atomic_add(hist_ptr + r0l, tl.zeros([R0N], tl.int32),
                             sem="relaxed", scope="cta")
        pre0 = tl.cumsum(cnt0, axis=0) - cnt0          # count of elements < b
        tb = tl.max(tl.where(pre0 <= n_cand - remain, r0l, -1))
        # Both counts over the [CAP] candidate block, packed into one
        # reduction (see the coarse select): counting per bin would be two
        # more 512-lane reductions for the same two numbers.
        pack = tl.sum(tl.where(live, (d0 > tb).to(tl.int32)
                               | ((d0 == tb).to(tl.int32) << 16), 0))
        n_eq = pack >> 16
        remain -= pack & 0xFFFF
        pref = pref | (tb.to(tl.uint32) << R0S)
        # ---- round 1: bits 10:0, only when the sub-bin overfills -----------
        # remain == n_eq means every element of the round-0 sub-bin is
        # admitted, so the low 11 bits never have to be ordered and the
        # cutoff is the sub-bin prefix itself (93% of rows at the GLM
        # prefill shape, 100% of the decode ones).
        full = remain == n_eq
        # One slot left in the cutoff sub-bin: the largest active key IS the
        # exact cutoff, so a single [CAP] max replaces the whole round-1
        # histogram (the firing row's remain is often 1).
        if (full == 0) & (remain == 1):
            actm = live & ((key_b >> R0S) == (pref >> R0S))
            pref = tl.max(tl.where(actm, key_b, 0))
        # With two entries left, two max reductions are exact as well. If the
        # maximum occurs twice, it is the cutoff; otherwise the second
        # distinct maximum is the cutoff.
        if (full == 0) & (remain == 2):
            act2 = live & ((key_b >> R0S) == (pref >> R0S))
            max1 = tl.max(tl.where(act2, key_b, 0))
            nmax1 = tl.sum((act2 & (key_b == max1)).to(tl.int32))
            max2 = tl.max(tl.where(act2 & (key_b < max1), key_b, 0))
            pref = tl.where(nmax1 >= 2, max1, max2)
        if (full == 0) & (remain > 2):
            # the active set must share every already-fixed bit, i.e. bits
            # >= 11. A wider window would let in the neighbouring digit and
            # move the cutoff up by one rank block.
            act = live & ((key_b >> R0S) == (pref >> R0S))
            # 11 bits -> the full BINS width; unlike round 0 this one is
            # skipped for almost every row, so its width is not worth a
            # dispatch of its own.
            rl1 = tl.arange(0, R1N)
            tl.debug_barrier()
            tl.store(hist_ptr + rl1, 0)
            tl.debug_barrier()
            tl.atomic_add(hist_ptr + (key_b & (R1N - 1)), cones, mask=act,
                          sem="relaxed", scope="cta")
            tl.debug_barrier()
            cnt1 = tl.atomic_add(hist_ptr + rl1, tl.zeros([R1N], tl.int32),
                                 sem="relaxed", scope="cta")
            pre1 = tl.cumsum(cnt1, axis=0) - cnt1
            tot = tl.sum(cnt1)
            # largest b with count(d >= b) >= remain <=> pre(b) <= tot - remain
            tb1 = tl.max(tl.where(pre1 <= tot - remain, rl1, -1))
            remain -= tl.sum(tl.where(rl1 > tb1, cnt1, 0))
            pref = pref | tb1.to(tl.uint32)
            # REQUIRED before the next histogram clear: without it a fast
            # warp clears bins a slow warp is still reading back, and the
            # row silently loses selected groups.
            tl.debug_barrier()

        cut_key = pref  # bits 31:0: shared high bits | refined low 21 bits
        # skipped round 1: the whole sub-bin is in, so admit ties with it and
        # leave the tie budget at zero
        tie_budget = tl.where(full, 0, remain)

        # ---------------- emit the staged candidates ------------------------
        # ctr[0] now only numbers the candidates (it starts at zero, the
        # selected set no longer draws from it); ctr[3] rations the ties.
        # When round 1 was skipped the sub-bin's members are distinct keys that
        # are all admitted, so `key >= cut_key` (not `>`) is the right test.
        sel = live & ((key_b > cut_key) | ((key_b == cut_key) & full))
        tie = live & (key_b == cut_key) & (full == 0)
        tpos = tl.atomic_add(ctr_ptr + 3 + tl.zeros([CAP], tl.int32), cones,
                             mask=tie, sem="relaxed", scope="cta")
        tie_ok = tie & (tpos < tie_budget)
        # `sel` and `tie_ok` are disjoint and together fill exactly
        # [count_above, count_above + remain0), and the order of selected
        # groups within a row is unspecified -- so both draw positions from the
        # same counter and leave in one emit instead of two (a second
        # [CAP]-wide position atomic plus a second 4-column store, ~1.4 us on
        # the two-row shapes).
        take = sel | tie_ok
        pos = tl.atomic_add(ctr_ptr + tl.zeros([CAP], tl.int32), cones,
                            mask=take, sem="relaxed", scope="cta")
        _emit(row_out, count_above + pos, idx_b, take, off, table_base,
              HAS_PT, HAS_OFF, POOL, POOL_PAD, DIG)
    else:
        # ---------------- overflow fallback: exact full-row rescan ----------
        # (no tl.cumsum here: two scans with differing layouts in one loop
        #  break ConvertTritonGPUToLLVM on this backend; smem atomics are also
        #  the cheaper way to place output positions anyway)
        if SAMP > 0:
            # This path needs the canonical bin, and the merged pass no longer
            # keeps a histogram to read it from: clear the bins (the sample
            # left them dirty) and rebuild it in one backward pass. Paid only
            # by the rows that miss the fast path (a mispredicted sample, or a
            # bin wider than the capacity), where the rescan reads below
            # dominate the extra pass anyway.
            for c in tl.range(0, tl.cdiv(BINS, BLOCK)):
                bins_off = c * BLOCK + lane
                tl.store(hist_ptr + bins_off, 0, mask=bins_off < BINS)
            tl.debug_barrier()
            for t in tl.range(0, n_tiles):
                idx = (n_tiles - 1 - t) * BLOCK + lane
                m = idx < length
                x = tl.load(row_score + idx, mask=m, other=0.0)
                d = (tl.where(m, _okey(x), 0) >> SHIFT0) & (BINS - 1)
                tl.atomic_add(hist_ptr + d, ones, mask=m, sem="relaxed",
                              scope="cta")
            tl.debug_barrier()
            if HISTN == BINS:
                hcnt_l = tle.gpu.metax.local_load(s_hist)
            else:
                hcnt_l = tl.atomic_add(hist_ptr + hbins,
                                       tl.zeros([BINS], tl.int32),
                                       sem="relaxed", scope="cta")
            pre_l = tl.cumsum(hcnt_l, axis=0) - hcnt_l
            t_l = tl.max(tl.where(pre_l <= length - group_topk, hbins, -1))
            pack_l = tl.sum(tl.where(hbins > t_l, hcnt_l, 0)
                            | (tl.minimum(tl.where(hbins == t_l, hcnt_l, 0),
                                          0x7FFF) << 16))
            rem_l = group_topk - (pack_l & 0xFFFF)
        else:
            t_l = t_bin
            rem_l = remain0
        cut_key, count_gt, remain = _rescan_refine(
            row_score, length, n_tiles, t_l, rem_l, SHIFT0, BLOCK,
                BINS)

        for t in tl.range(0, n_tiles):
            idx = t * BLOCK + lane
            m = idx < length
            x = tl.load(row_score + idx, mask=m, other=float("-inf"))
            key = _okey(x)
            d = (tl.where(m, key, 0) >> SHIFT0) & (BINS - 1)
            in_bin = (d == t_l) & m
            above = ((d > t_l) & m) | (in_bin & (key > cut_key))
            tie = in_bin & (key == cut_key)
            pos_a = tl.atomic_add(ctr_ptr + tl.zeros([BLOCK], tl.int32), ones,
                                  mask=above, sem="relaxed", scope="cta")
            _emit_one(row_out, pos_a, idx, above, off, table_base,
                      HAS_PT, HAS_OFF, POOL)
            tpos = tl.atomic_add(ctr_ptr + 3 + tl.zeros([BLOCK], tl.int32),
                                 ones, mask=tie, sem="relaxed", scope="cta")
            tie_ok = tie & (tpos < remain)
            pos_t = tl.atomic_add(ctr_ptr + tl.zeros([BLOCK], tl.int32), ones,
                                  mask=tie_ok, sem="relaxed", scope="cta")
            _emit_one(row_out, pos_t, idx, tie_ok, off, table_base,
                      HAS_PT, HAS_OFF, POOL)

    # ---------------- tail + padding ---------------------------------------
    if HAS_TAIL:
        s = tl.arange(0, POOL_PAD)
        col = token_topk + s
        mtail = s < tail_n
        raw_tail = length * POOL + s
        if HAS_PT:
            val = tl.load(table_base + raw_tail, mask=mtail, other=-1)
        elif HAS_OFF:
            val = raw_tail + off
        else:
            val = raw_tail
        tl.store(row_out + col, val, mask=mtail)
        # columns [token_topk + tail_n, out_cols) must be -1; out_cols is
        # token_topk + POOL - 1, so never write past s == POOL - 1.
        pad = (s >= tail_n) & (s < POOL - 1)
        tl.store(row_out + col, tl.full((POOL_PAD,), -1, tl.int32), mask=pad)


# ---------------------------------------------------------------------------
# split path: one row across NSEG CTAs (wide, one-wave rows)
# ---------------------------------------------------------------------------
#
# A row that is much wider than one 512-thread CTA can stream leaves most of
# the 104 APs idle when batch is small (B=32 -> 32 CTAs, B=2 -> 2).  The
# per-element passes dominate there, so the row is split across NSEG CTAs in
# four launches:
#
#   K1  partial 2048-bin histograms -> global scratch     (grid B*NSEG)
#   K2  merge the NSEG histograms, pick t_bin/counts      (grid B)
#   K3  classify + stage each segment's two slot lists    (grid B*NSEG)
#   K4  concatenate the segment lists, refine, emit       (grid B)
#
# A kernel boundary is a full memory fence, so no cross-CTA handshake is
# needed.  Only K1 pays the fixed 2048-bin histogram cost (~4.5 us); K3 only
# carries two counters and two small lists, which is what makes the split pay
# at all (a per-segment histogram in K3 would not).  The single-CTA kernel
# keeps every other shape.


@triton.jit
def _split_pre_kernel(score_ptr, lengths_ptr, starts_ptr, meta_ptr,
                      width, group_topk,
                      HAS_STARTS: tl.constexpr, BINS: tl.constexpr,
                      SHIFT0: tl.constexpr, BLOCK: tl.constexpr):
    """One CTA per row: full-row histogram + threshold, straight into meta.

    This is the fused form of K1+K2 for rows that are not wide enough for the
    per-segment histogram to pay (W=8192): it avoids the global histogram
    round-trip.  Rows with length <= group_topk are left to K4's identity path
    and get no meta entry.
    """
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row)
    if length <= group_topk:
        return
    start = tl.load(starts_ptr + row) if HAS_STARTS else 0
    row_score = score_ptr + row.to(tl.int64) * width + start.to(tl.int64)
    lane = tl.arange(0, BLOCK)
    ones = tl.full([BLOCK], 1, tl.int32)
    hbins = tl.arange(0, BINS)
    s_hist = tle.gpu.alloc([BINS], dtype=tl.int32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    hist_ptr = tle.gpu.local_ptr(s_hist, (0,))
    for c in tl.range(0, tl.cdiv(BINS, BLOCK)):
        bo = c * BLOCK + lane
        tl.store(hist_ptr + bo, 0, mask=bo < BINS)
    tl.debug_barrier()
    n_tiles = tl.cdiv(length, BLOCK)
    for t in tl.range(0, n_tiles):
        idx = (n_tiles - 1 - t) * BLOCK + lane
        m = idx < length
        x = tl.load(row_score + idx, mask=m, other=0.0)
        d = (tl.where(m, _okey(x), 0) >> SHIFT0) & (BINS - 1)
        tl.atomic_add(hist_ptr + d, ones, mask=m, sem="relaxed", scope="cta")
    tl.debug_barrier()
    hcnt = tle.gpu.metax.local_load(s_hist)
    pre = tl.cumsum(hcnt, axis=0) - hcnt
    t_bin = tl.max(tl.where(pre <= length - group_topk, hbins, -1))
    pack = tl.sum(tl.where(hbins > t_bin, hcnt, 0)
                  | (tl.minimum(tl.where(hbins == t_bin, hcnt, 0),
                                0x7FFF) << 16))
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 0, t_bin)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 1, pack & 0xFFFF)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 2, pack >> 16)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 3, 0)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 4, 0)


@triton.jit
def _split_hist_kernel(score_ptr, lengths_ptr, starts_ptr, phist_ptr,
                       width,
                       NSEG: tl.constexpr, HAS_STARTS: tl.constexpr,
                       BINS: tl.constexpr, SHIFT0: tl.constexpr,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NSEG
    seg = pid % NSEG
    length = tl.load(lengths_ptr + row)
    start = tl.load(starts_ptr + row) if HAS_STARTS else 0
    row_score = score_ptr + row.to(tl.int64) * width + start.to(tl.int64)
    chunk = tl.cdiv(length, NSEG)
    lo = seg * chunk
    hi = tl.minimum(lo + chunk, length)

    lane = tl.arange(0, BLOCK)
    ones = tl.full([BLOCK], 1, tl.int32)
    hbins = tl.arange(0, BINS)
    s_hist = tle.gpu.alloc([BINS], dtype=tl.int32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    hist_ptr = tle.gpu.local_ptr(s_hist, (0,))
    for c in tl.range(0, tl.cdiv(BINS, BLOCK)):
        bo = c * BLOCK + lane
        tl.store(hist_ptr + bo, 0, mask=bo < BINS)
    tl.debug_barrier()

    n_tiles = tl.cdiv(hi - lo, BLOCK)
    for t in tl.range(0, n_tiles):
        idx = lo + t * BLOCK + lane
        m = idx < hi
        x = tl.load(row_score + idx, mask=m, other=0.0)
        d = (tl.where(m, _okey(x), 0) >> SHIFT0) & (BINS - 1)
        tl.atomic_add(hist_ptr + d, ones, mask=m, sem="relaxed", scope="cta")
    tl.debug_barrier()
    hcnt = tle.gpu.metax.local_load(s_hist)
    tl.store(phist_ptr + pid.to(tl.int64) * BINS + hbins, hcnt)


@triton.jit
def _split_merge_kernel(phist_ptr, lengths_ptr, meta_ptr,
                        group_topk,
                        NSEG: tl.constexpr, BINS: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row)
    hbins = tl.arange(0, BINS)
    hcnt = tl.zeros([BINS], tl.int32)
    for s in tl.static_range(NSEG):
        hcnt += tl.load(phist_ptr + (row * NSEG + s).to(tl.int64) * BINS
                        + hbins)
    pre = tl.cumsum(hcnt, axis=0) - hcnt
    t_bin = tl.max(tl.where(pre <= length - group_topk, hbins, -1))
    # same saturation as the single-CTA path: the tie-bin field must not reach
    # the sign bit, or `n_cand` reads back small and the guard lies.
    pack = tl.sum(tl.where(hbins > t_bin, hcnt, 0)
                  | (tl.minimum(tl.where(hbins == t_bin, hcnt, 0),
                                0x7FFF) << 16))
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 0, t_bin)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 1, pack & 0xFFFF)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 2, pack >> 16)
    # K3 hands out slot bases with one global atomic per segment, so the two
    # per-row counters have to start at zero for this call.
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 3, 0)
    tl.store(meta_ptr + row.to(tl.int64) * 5 + 4, 0)


@triton.jit
def _split_classify_kernel(score_ptr, lengths_ptr, starts_ptr, meta_ptr,
                           sel_all_ptr, cand_all_ptr,
                           width,
                           NSEG: tl.constexpr, HAS_STARTS: tl.constexpr,
                           BINS: tl.constexpr, SHIFT0: tl.constexpr,
                           group_topk,
                           CAP: tl.constexpr, CAPP: tl.constexpr,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // NSEG
    seg = pid % NSEG
    length = tl.load(lengths_ptr + row)
    # Rows shorter than the selection are handled by K4's identity path and
    # never got a meta entry from the pre-pass.
    if length <= group_topk:
        return
    start = tl.load(starts_ptr + row) if HAS_STARTS else 0
    row_score = score_ptr + row.to(tl.int64) * width + start.to(tl.int64)
    t_bin = tl.load(meta_ptr + row.to(tl.int64) * 5 + 0)
    chunk = tl.cdiv(length, NSEG)
    lo = seg * chunk
    hi = tl.minimum(lo + chunk, length)

    lane = tl.arange(0, BLOCK)
    ones = tl.full([BLOCK], 1, tl.int32)
    s_ctr = tle.gpu.alloc([4], dtype=tl.int32, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ctr_ptr = tle.gpu.local_ptr(s_ctr, (0,))
    s_sel = tle.gpu.alloc([CAPP], dtype=tl.uint16, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    sel_ptr = tle.gpu.local_ptr(s_sel, (0,))
    s_cand = tle.gpu.alloc([CAP], dtype=tl.uint16, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    cand_ptr = tle.gpu.local_ptr(s_cand, (0,))
    tl.store(ctr_ptr + tl.arange(0, 4), 0)
    tl.debug_barrier()

    n_tiles = tl.cdiv(hi - lo, BLOCK)
    for t in tl.range(0, n_tiles):
        idx = lo + t * BLOCK + lane
        m = idx < hi
        x = tl.load(row_score + idx, mask=m, other=0.0)
        key = _okey(x)
        d = (tl.where(m, key, 0) >> SHIFT0) & (BINS - 1)
        eq = m & (d == t_bin)
        keep = m & (d >= t_bin)
        pos = tl.atomic_add(ctr_ptr + 1 + eq.to(tl.int32), ones,
                            mask=keep, sem="relaxed", scope="cta")
        tl.store(tl.where(eq, cand_ptr + pos, sel_ptr + pos),
                 idx.to(tl.uint16), mask=keep)
    tl.debug_barrier()

    # Read the two counters with a BLOCK-wide masked atomic (a narrow smem
    # access does not lower on this backend) and pack them in one reduction.
    # A plain load is not usable here: the backend does not model the relaxed
    # smem atomics as writes, so it would be hoisted (the histogram bug).
    cvals = tl.atomic_add(ctr_ptr + lane, tl.zeros([BLOCK], tl.int32),
                          mask=lane < 4, sem="relaxed", scope="cta")
    packed_cnt = tl.sum(tl.where(lane == 1, cvals, 0)
                        | (tl.where(lane == 2, cvals, 0) << 16))
    sel_cnt = packed_cnt & 0xFFFF
    cand_cnt = packed_cnt >> 16
    # One global atomic per segment reserves this segment's slice of the
    # row's concatenated list; the per-row counter was zeroed by K2.
    base_s = tl.atomic_add(meta_ptr + row.to(tl.int64) * 5 + 3, sel_cnt)
    base_c = tl.atomic_add(meta_ptr + row.to(tl.int64) * 5 + 4, cand_cnt)
    lp = tl.arange(0, CAPP)
    lv = tl.load(sel_ptr + lp, mask=lp < sel_cnt, other=0).to(tl.int32)
    tl.store(sel_all_ptr + row.to(tl.int64) * CAPP + base_s + lp, lv,
             mask=lp < sel_cnt)
    cp = tl.arange(0, CAP)
    cv = tl.load(cand_ptr + cp, mask=cp < cand_cnt, other=0).to(tl.int32)
    tl.store(cand_all_ptr + row.to(tl.int64) * CAP + base_c + cp, cv,
             mask=cp < cand_cnt)


@triton.jit
def _split_finish_kernel(score_ptr, lengths_ptr, out_ptr, page_table_ptr,
                         offset_ptr, seq_ptr, trow_ptr, starts_ptr,
                         meta_ptr, sel_all_ptr, cand_all_ptr,
                         width, out_cols, table_width, token_topk,
                         group_topk,
                         NSEG: tl.constexpr, POOL: tl.constexpr,
                         POOL_PAD: tl.constexpr, HAS_PT: tl.constexpr,
                         HAS_OFF: tl.constexpr, HAS_TAIL: tl.constexpr,
                         HAS_STARTS: tl.constexpr, HAS_TROW: tl.constexpr,
                         BINS: tl.constexpr, SHIFT0: tl.constexpr,
                         HISTN: tl.constexpr,
                         R0S: tl.constexpr, R0N: tl.constexpr,
                         R1N: tl.constexpr,
                         CAP: tl.constexpr, CAPP: tl.constexpr,
                         BLOCK: tl.constexpr, ETILE: tl.constexpr):
    row = tl.program_id(0)
    length = tl.load(lengths_ptr + row)
    start = tl.load(starts_ptr + row) if HAS_STARTS else 0
    off = tl.load(offset_ptr + row) if HAS_OFF else 0
    tail_n = tl.load(seq_ptr + row) % POOL if HAS_TAIL else 0
    row_score = score_ptr + row.to(tl.int64) * width + start.to(tl.int64)
    row_out = out_ptr + row.to(tl.int64) * out_cols
    table_base = (
        page_table_ptr
        + (tl.load(trow_ptr + row) if HAS_TROW else row).to(tl.int64)
        * table_width
    )
    lane = tl.arange(0, BLOCK)
    ones = tl.full([BLOCK], 1, tl.int32)

    # ---------------- early path: length <= group_topk (identity) -----------
    if length <= group_topk:
        limit = length * POOL + tail_n
        for c0 in range(0, out_cols, BLOCK):
            col = c0 + lane
            m = col < out_cols
            live = col < limit
            if HAS_PT:
                val = tl.load(table_base + col, mask=live, other=-1)
            elif HAS_OFF:
                val = col + off
            else:
                val = col
            val = tl.where(live, val, -1)
            tl.store(row_out + col, val, mask=m)
        return

    t_bin = tl.load(meta_ptr + row.to(tl.int64) * 5 + 0)
    count_above = tl.load(meta_ptr + row.to(tl.int64) * 5 + 1)
    n_cand = tl.load(meta_ptr + row.to(tl.int64) * 5 + 2)
    remain0 = group_topk - count_above

    s_hist = tle.gpu.alloc([HISTN], dtype=tl.int32, layout=None,
                           scope=tle.gpu.smem, nv_mma_shared_layout=False)
    hist_ptr = tle.gpu.local_ptr(s_hist, (0,))
    s_ctr = tle.gpu.alloc([4], dtype=tl.int32, layout=None,
                          scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ctr_ptr = tle.gpu.local_ptr(s_ctr, (0,))
    tl.store(ctr_ptr + tl.arange(0, 4), 0)
    tl.debug_barrier()

    if n_cand <= CAP:
        # ---------------- emit the selected set -----------------------------
        plane = tl.arange(0, CAPP)
        live_ab = plane < count_above
        idx_a = tl.load(sel_all_ptr + row.to(tl.int64) * CAPP + plane,
                        mask=live_ab, other=0).to(tl.int32)
        _emit(row_out, plane, idx_a, live_ab, off, table_base,
              HAS_PT, HAS_OFF, POOL, POOL_PAD, ETILE)

        # ---------------- exact radix refine over the candidates ------------
        clane = tl.arange(0, CAP)
        live = clane < n_cand
        cones = tl.full([CAP], 1, tl.int32)
        idx_b = tl.load(cand_all_ptr + row.to(tl.int64) * CAP + clane,
                        mask=live, other=0).to(tl.int32)
        key_b = _okey(tl.load(row_score + idx_b, mask=live, other=0.0))

        r0l = tl.arange(0, R0N)
        pref = t_bin.to(tl.uint32) << SHIFT0
        remain = remain0
        d0 = (key_b >> R0S) & (R0N - 1)
        tl.store(hist_ptr + r0l, 0)
        tl.debug_barrier()
        tl.atomic_add(hist_ptr + d0, cones, mask=live,
                      sem="relaxed", scope="cta")
        tl.debug_barrier()
        cnt0 = tl.atomic_add(hist_ptr + r0l, tl.zeros([R0N], tl.int32),
                             sem="relaxed", scope="cta")
        pre0 = tl.cumsum(cnt0, axis=0) - cnt0
        tb = tl.max(tl.where(pre0 <= n_cand - remain, r0l, -1))
        packr = tl.sum(tl.where(live, (d0 > tb).to(tl.int32)
                                | ((d0 == tb).to(tl.int32) << 16), 0))
        n_eq = packr >> 16
        remain -= packr & 0xFFFF
        pref = pref | (tb.to(tl.uint32) << R0S)
        full = remain == n_eq
        # One slot left in the cutoff sub-bin: the largest active key IS the
        # exact cutoff, so a single [CAP] max replaces the whole round-1
        # histogram (the firing row's remain is often 1).
        if (full == 0) & (remain == 1):
            actm = live & ((key_b >> R0S) == (pref >> R0S))
            pref = tl.max(tl.where(actm, key_b, 0))
        # With two entries left, two max reductions are exact as well. If the
        # maximum occurs twice, it is the cutoff; otherwise the second
        # distinct maximum is the cutoff.
        if (full == 0) & (remain == 2):
            act2 = live & ((key_b >> R0S) == (pref >> R0S))
            max1 = tl.max(tl.where(act2, key_b, 0))
            nmax1 = tl.sum((act2 & (key_b == max1)).to(tl.int32))
            max2 = tl.max(tl.where(act2 & (key_b < max1), key_b, 0))
            pref = tl.where(nmax1 >= 2, max1, max2)
        if (full == 0) & (remain > 2):
            act = live & ((key_b >> R0S) == (pref >> R0S))
            rl1 = tl.arange(0, R1N)
            tl.debug_barrier()
            tl.store(hist_ptr + rl1, 0)
            tl.debug_barrier()
            tl.atomic_add(hist_ptr + (key_b & (R1N - 1)), cones, mask=act,
                          sem="relaxed", scope="cta")
            tl.debug_barrier()
            cnt1 = tl.atomic_add(hist_ptr + rl1, tl.zeros([R1N], tl.int32),
                                 sem="relaxed", scope="cta")
            pre1 = tl.cumsum(cnt1, axis=0) - cnt1
            tot = tl.sum(cnt1)
            tb1 = tl.max(tl.where(pre1 <= tot - remain, rl1, -1))
            remain -= tl.sum(tl.where(rl1 > tb1, cnt1, 0))
            pref = pref | tb1.to(tl.uint32)
            tl.debug_barrier()
        cut_key = pref
        tie_budget = tl.where(full, 0, remain)

        sel = live & ((key_b > cut_key) | ((key_b == cut_key) & full))
        tie = live & (key_b == cut_key) & (full == 0)
        tpos = tl.atomic_add(ctr_ptr + 3 + tl.zeros([CAP], tl.int32), cones,
                             mask=tie, sem="relaxed", scope="cta")
        tie_ok = tie & (tpos < tie_budget)
        take = sel | tie_ok
        pos = tl.atomic_add(ctr_ptr + tl.zeros([CAP], tl.int32), cones,
                            mask=take, sem="relaxed", scope="cta")
        _emit(row_out, count_above + pos, idx_b, take, off, table_base,
              HAS_PT, HAS_OFF, POOL, POOL_PAD, ETILE)
    else:
        # ---------------- overflow fallback: exact full-row rescan ----------
        n_tiles = tl.cdiv(length, BLOCK)
        cut_key, count_gt, remain = _rescan_refine(
            row_score, length, n_tiles, t_bin, remain0, SHIFT0, BLOCK, BINS)
        for t in tl.range(0, n_tiles):
            idx = t * BLOCK + lane
            m = idx < length
            x = tl.load(row_score + idx, mask=m, other=float("-inf"))
            key = _okey(x)
            d = (tl.where(m, key, 0) >> SHIFT0) & (BINS - 1)
            in_bin = (d == t_bin) & m
            above = ((d > t_bin) & m) | (in_bin & (key > cut_key))
            tie = in_bin & (key == cut_key)
            pos_a = tl.atomic_add(ctr_ptr + tl.zeros([BLOCK], tl.int32), ones,
                                  mask=above, sem="relaxed", scope="cta")
            _emit_one(row_out, pos_a, idx, above, off, table_base,
                      HAS_PT, HAS_OFF, POOL)
            tpos = tl.atomic_add(ctr_ptr + 3 + tl.zeros([BLOCK], tl.int32),
                                 ones, mask=tie, sem="relaxed", scope="cta")
            tie_ok = tie & (tpos < remain)
            pos_t = tl.atomic_add(ctr_ptr + tl.zeros([BLOCK], tl.int32), ones,
                                  mask=tie_ok, sem="relaxed", scope="cta")
            _emit_one(row_out, pos_t, idx, tie_ok, off, table_base,
                      HAS_PT, HAS_OFF, POOL)

    # ---------------- tail + padding ---------------------------------------
    if HAS_TAIL:
        s = tl.arange(0, POOL_PAD)
        col = token_topk + s
        mtail = s < tail_n
        raw_tail = length * POOL + s
        if HAS_PT:
            val = tl.load(table_base + raw_tail, mask=mtail, other=-1)
        elif HAS_OFF:
            val = raw_tail + off
        else:
            val = raw_tail
        tl.store(row_out + col, val, mask=mtail)
        pad = (s >= tail_n) & (s < POOL - 1)
        tl.store(row_out + col, tl.full((POOL_PAD,), -1, tl.int32), mask=pad)


# ---------------------------------------------------------------------------
# host wrapper
# ---------------------------------------------------------------------------


# C500 AP count: one wave of one program per AP. Above this the tiled path is
# better off keeping two CTAs resident per AP.
ONE_WAVE = 104


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Fast launch path
#
# `JITFunction.run` costs 24-27 us of host time for this 30-argument kernel
# (argument binding + specialization, cache key, hooks, metadata), while the
# compiled kernel's own launcher costs ~3 us.  The <=1024-lane shapes have
# 15-21 us of *device* work, so on them the wrapper -- not the kernel -- is
# what a per-call timer measures.  This caches the CompiledKernel that the JIT
# path returns and re-launches it directly, substituting the tensor arguments
# of the current call.
#
# Every value that decides which binary is launched is part of the key (dtype,
# batch, row length, row stride, out_cols, table stride, token/group topk, the
# HAS_* flags, block/bins/shift0/cap/num_warps/num_stages): two calls with the
# same key bind the same kernel with the same scalar arguments and constexprs,
# so only the eight pointers can differ.  The entry is installed only when all
# eight pointers are 16-byte aligned, so the pointer-alignment specialization
# the stored binary was compiled with also holds for every later hit; anything
# else falls back to the JIT path.
# ---------------------------------------------------------------------------
_LAUNCH_CACHE: dict = {}
_TENSOR_ARGS = ("score_ptr", "lengths_ptr", "out_ptr", "page_table_ptr",
                "offset_ptr", "starts_ptr", "seq_ptr", "trow_ptr")
_DIST_LITE = None

# Scratch for the split path, keyed by (device, batch, NSEG, group_topk, cap,
# capp).  Reused across calls: the benchmark captures the wrapper into a CUDA
# graph after warmup, and an allocation during capture would be unsafe.
_SPLIT_CACHE: dict = {}


def _split_topk(score, lengths, output, pool_size, page_table,
                topk_indices_offset, row_starts, seq_lens,
                page_table_row_index, width, out_cols, table_width,
                token_topk, group_topk, nseg, block, cap, capp,
                r0s, r0n, r1n, pre_pass=0) -> None:
    """One row across `nseg` CTAs: partial hist -> merge -> classify -> finish."""
    batch = score.shape[0]
    dev = score.device
    key = (dev.index if dev.index is not None else -1, batch, nseg,
           group_topk, cap, capp, pre_pass)
    ent = _SPLIT_CACHE.get(key)
    if ent is None:
        n = batch * nseg
        ent = (
            None if pre_pass else
            torch.empty((n, 2048), dtype=torch.int32, device=dev),
            torch.empty((batch, 5), dtype=torch.int32, device=dev),
            torch.empty((batch, capp), dtype=torch.int32, device=dev),
            torch.empty((batch, cap), dtype=torch.int32, device=dev),
        )
        _SPLIT_CACHE[key] = ent
    phist, meta, sell, candl = ent
    so = row_starts if row_starts is not None else lengths
    po = page_table if page_table is not None else score
    oo = topk_indices_offset if topk_indices_offset is not None else lengths
    qo = seq_lens if seq_lens is not None else lengths
    to = page_table_row_index if page_table_row_index is not None else lengths

    if pre_pass:
        _split_pre_kernel[(batch,)](
            score, lengths, so, meta, width, group_topk,
            HAS_STARTS=row_starts is not None, BINS=2048, SHIFT0=21,
            BLOCK=block, num_warps=8)
    else:
        _split_hist_kernel[(batch * nseg,)](
            score, lengths, so, phist, width,
            NSEG=nseg, HAS_STARTS=row_starts is not None, BINS=2048,
            SHIFT0=21, BLOCK=block, num_warps=8)
        _split_merge_kernel[(batch,)](
            phist, lengths, meta, group_topk,
            NSEG=nseg, BINS=2048, num_warps=8)
    _split_classify_kernel[(batch * nseg,)](
        score, lengths, so, meta, sell, candl, width,
        NSEG=nseg, HAS_STARTS=row_starts is not None, BINS=2048, SHIFT0=21,
        group_topk=group_topk, CAP=cap, CAPP=capp, BLOCK=block,
        num_warps=8)
    _split_finish_kernel[(batch,)](
        score, lengths, output, po, oo, qo, to, so,
        meta, sell, candl,
        width, out_cols, table_width, token_topk, group_topk,
        NSEG=nseg, POOL=pool_size, POOL_PAD=_next_pow2(pool_size),
        HAS_PT=page_table is not None, HAS_OFF=topk_indices_offset is not None,
        HAS_TAIL=seq_lens is not None, HAS_STARTS=row_starts is not None,
        HAS_TROW=page_table_row_index is not None, BINS=2048, SHIFT0=21,
        HISTN=max(2048, r0n),
        R0S=r0s, R0N=r0n, R1N=r1n,
        CAP=cap, CAPP=capp, BLOCK=block, ETILE=(1 if batch <= 2 else 0),
        num_warps=8)


def _dist_lite() -> bool:
    """True when the JIT path would pass distributed-runtime launch args."""
    global _DIST_LITE
    if _DIST_LITE is None:
        try:
            from triton.runtime._distributed import DistributedRtContext
            _DIST_LITE = bool(DistributedRtContext().is_lite_mode)
        except Exception:  # noqa: BLE001 - unknown -> never take the fast path
            _DIST_LITE = True
    return _DIST_LITE


def _hooks_idle() -> bool:
    """True when no launch hook would observe the launch.

    The hooks are `HookChain` objects, not None, once any tool has asked for
    them; an empty chain does nothing, so it counts as idle and the direct
    launch can skip building the launch metadata the JIT path would build.
    """
    for hook in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if hook is None:
            continue
        calls = getattr(hook, "calls", True)
        if calls:
            return False
    return True


def _direct_launch(key, tensors, grid0) -> bool:
    """Re-launch the cached kernel for `key`; False means "use the JIT path"."""
    ent = _LAUNCH_CACHE.get(key)
    if ent is None or _DIST_LITE:
        return False
    if not _hooks_idle():
        return False
    run, fn, pk, tpl, slot, get_stream, dev = ent
    ptr = 0
    for t in tensors:
        ptr |= t.data_ptr()
    if ptr & 15:  # alignment specialization of the stored binary no longer holds
        return False
    args = list(tpl)
    for i, t in zip(slot, tensors):
        args[i] = t
    run(grid0, 1, 1, get_stream(dev), fn, pk, None, None, None, *args)
    return True


# ---------------------------------------------------------------------------
# Lean re-entry
#
# `_direct_launch` above already avoids the JIT path, but everything in front of
# it still runs on every call: the dtype/stride checks, the `wide` band
# selection (with its `_fits` loop), the 14-element `kargs` and the 24-element
# `lkey`, plus the pointer walk.  Measured on this device that prelude costs
# ~6 us of host time, which is *all* of the small shapes: an empty kernel on
# grid<=32 costs 13.5 us (probes/probe_v4_floor.py) while decode_3ki's real
# kernel costs 13.09 us, so there is nothing left in the kernel to remove.
# This cache answers the same question from the call's shapes and flags alone
# and re-launches in place, so a repeat call never reaches the prologue.
# ---------------------------------------------------------------------------
_FAST_LAUNCH: dict = {}


def _fast_key(score, lengths, output, pool_size, page_table, topk_indices_offset,
              row_starts, seq_lens, page_table_row_index, block, bins, shift0,
              cap, num_warps, num_stages):
    """Everything a launch depends on except the eight pointers.

    Shapes are ``torch.Size`` (a C tuple) so this is a handful of attribute
    reads and one tuple hash -- cheap enough to sit in front of the prologue.
    """
    # Built from ints, not torch.Size: indexing a shape is a C int read, while
    # hashing the Size objects themselves measured 1.06 us per call.
    return (score.shape[0], score.shape[1], score.stride(0), output.shape[1],
            score.dtype, lengths.dtype, output.dtype, pool_size,
            -1 if page_table is None else page_table.stride(0),
            topk_indices_offset is not None, row_starts is not None,
            seq_lens is not None, page_table_row_index is not None,
            block, bins, shift0, cap, num_warps, num_stages)


def _fast_launch(key, score, lengths, output, page_table, topk_indices_offset,
                 row_starts, seq_lens, page_table_row_index) -> bool:
    """Re-launch a cached binary for `key`; False means "run the full path"."""
    ent = _FAST_LAUNCH.get(key)
    if ent is None:
        return False
    run, fn, pk, args, quick, slot, get_stream, dev = ent
    po = score if page_table is None else page_table
    oo = lengths if topk_indices_offset is None else topk_indices_offset
    so = lengths if row_starts is None else row_starts
    qo = lengths if seq_lens is None else seq_lens
    to = lengths if page_table_row_index is None else page_table_row_index
    # the cached binary was compiled for 16-byte-aligned pointers
    if ((score.data_ptr() | lengths.data_ptr() | output.data_ptr() | po.data_ptr()
         | oo.data_ptr() | so.data_ptr() | qo.data_ptr() | to.data_ptr()) & 15):
        return False
    if quick:
        args[0:8] = (score, lengths, output, po, oo, so, qo, to)
    else:                                   # non-contiguous slots: patch one by one
        args[slot[0]] = score
        args[slot[1]] = lengths
        args[slot[2]] = output
        args[slot[3]] = po
        args[slot[4]] = oo
        args[slot[5]] = so
        args[slot[6]] = qo
        args[slot[7]] = to
    run(score.shape[0], 1, 1, get_stream(dev), fn, pk, None, None, None, *args)
    return True


def _cache_fast_launch(key, kern, kargs, kkw) -> None:
    """Install the lean re-entry for `key` (same binder the JIT path uses)."""
    if kern is None or key in _FAST_LAUNCH or _dist_lite():
        return
    if not _hooks_idle():
        return
    try:
        from triton.runtime import driver
        dev = driver.active.get_current_device()
        binder = kpool_topk_transform_kernel.device_caches[dev][4]
        params = binder(*kargs, **kkw)[0]
        names = list(params)
        slots = tuple(names.index(n) for n in _TENSOR_ARGS)
        # the eight tensor params are the leading arguments -> one slice store
        contiguous_slots = slots == (0, 1, 2, 3, 4, 5, 6, 7)
        args = list(params.values())
        _FAST_LAUNCH[key] = (kern.run, kern.function, kern.packed_metadata, args,
                             contiguous_slots, slots,
                             driver.active.get_current_stream, dev)
    except Exception:  # noqa: BLE001 - an unusable cache is not an error
        return


def _cache_launch(key, kern, kargs, kkw, grid0) -> None:
    """Store the compiled kernel + its bound argument template for `key`."""
    if kern is None or key in _LAUNCH_CACHE or _dist_lite():
        return
    if not _hooks_idle():
        return
    try:
        from triton.runtime import driver
        dev = driver.active.get_current_device()
        binder = kpool_topk_transform_kernel.device_caches[dev][4]
        params = binder(*kargs, **kkw)[0]
        names = list(params)
        slot = tuple(names.index(n) for n in _TENSOR_ARGS)
        tpl = list(params.values())
        if any(tpl[i].data_ptr() & 15 for i in slot):
            return
        _LAUNCH_CACHE[key] = (kern.run, kern.function, kern.packed_metadata,
                              tpl, slot, driver.active.get_current_stream, dev)
    except Exception:  # noqa: BLE001 - an unusable cache is not an error
        return


def fast_kpool_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_indices_offset: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    page_table_row_index: torch.Tensor | None = None,
    *,
    block: int = 4096,
    bins: int = 2048,
    shift0: int = 21,
    cap: int = 2048,
    num_warps: int = 8,
    num_stages: int = 3,
) -> None:
    """接口名（提交要求）：kpool_topk_transform 的融合算子 ABI（原地写 output）。"""
    # Lean re-entry: a repeat call with the same shapes/flags skips the whole
    # prologue (see `_fast_key`).  ~6 us of host time hangs on this on the
    # small shapes, where the kernel itself is already at the launch floor.
    fkey = _fast_key(score, lengths, output, pool_size, page_table,
                     topk_indices_offset, row_starts, seq_lens,
                     page_table_row_index, block, bins, shift0, cap,
                     num_warps, num_stages)
    if _FAST_LAUNCH and _fast_launch(fkey, score, lengths, output, page_table,
                                     topk_indices_offset, row_starts, seq_lens,
                                     page_table_row_index):
        return

    if score.dtype != torch.float32:
        raise NotImplementedError("score must be float32")
    if score.stride(-1) != 1:
        raise NotImplementedError("score last dim must be contiguous")
    if pool_size < 1:
        raise ValueError("pool_size must be >= 1")

    batch = score.shape[0]
    if batch == 0:
        return

    out_cols = output.shape[1]
    tail_cols = (pool_size - 1) if seq_lens is not None else 0
    token_topk = out_cols - tail_cols
    if token_topk % pool_size:
        raise ValueError("token_topk must be divisible by pool_size")
    group_topk = token_topk // pool_size

    width = score.stride(0)
    # floor at one thread-block: a block narrower than the thread count fails
    # to lower on this backend (same quirk as the narrow smem accesses)
    wide = max(_next_pow2(int(score.shape[1])), 512)
    # Shared-memory budget for the single-read digit staging. The backend adds
    # one [BLOCK] staging buffer (16 KB at the default block) for the pass-1
    # load loop on top of the declared allocations -- measured: a kernel whose
    # arrays sum to 51220 B reports "Required: 67600". A row of <= 16384 lanes
    # wants a 32 KB digit array, which only fits if the candidate capacity
    # shrinks, so the capacity is chosen together with the digit decision:
    # the hot shapes hold ~210 candidates in the threshold bin, so 1024 is
    # still five times the headroom, and a larger bin falls back to the exact
    # full-row rescan (correct, just slower).
    # Candidate capacity: the [CAP] region of `s_sel` is 4 KB of shared memory
    # at the default 2048 on top of everything else, and shared memory is what
    # limits the resident CTAs per AP.
    # Measured on prefill_63ki (16384-wide rows): cap 2048 -> 230.8 us
    # (26768 B, 2 CTAs/AP), 512 -> 217.8 us (14352 B, 4 CTAs/AP). The
    # 1024-lane rows want it too, for the opposite reason: they are not
    # occupancy-bound (one CTA per AP either way), the refine's [CAP]-wide
    # blocks are just fixed cost there. Device time measured through the
    # compiled kernel's launcher (the full JIT path is host-bound at ~24 us on
    # these shapes and hides all of it): decode_3ki 2048/1024/512 ->
    # 17.1/16.4/15.2 us, prefill_3ki -> 23.3/21.3/20.8 us. Below 512 the
    # block stops lowering (narrow-smem-access quirk).
    # Block width: one tile per row for the 8192-lane rows, and half as many
    # tiles for the 65536-lane ones.  Measured (+/- in us, probes/probe_v7_sweep):
    # B32 23.23->22.95, gK512 23.81->23.12, pool3 23.86->23.18, L65536
    # 85.00->78.08.  The 16383-lane rows must stay at 4096: there the wider tile
    # doubles the [BLOCK] staging buffer and pushes the row off the digit path
    # (decode_63ki 34.1 -> 62.7), a full grid wants the narrower tile for
    # occupancy (B256_W65536 228 -> 278 us at 8192), and the page-table mode
    # needs the shared memory it would take (page_table 23.6 -> 55.4 us), so it
    # is a one-wave, no-page-table rule.
    if ((4096 < wide <= 8192 or wide > 49152) and batch <= ONE_WAVE
            and page_table is None):
        block = 8192
    if wide <= 1024:
        # A 1024-lane row is register-resident and its tiled passes predicate
        # over BLOCK lanes, of which only `length` are live: at 4096 that is
        # 766 live lanes out of 8 per thread on prefill_3ki. Halving the block
        # halves the wasted issue (prefill_3ki 17.10 -> 16.05 us, decode_3ki
        # flat). 1024 is no better (prefill flat, decode_3ki 13.95), and the
        # 16k-lane shapes need the wide tile (decode_63ki 36.1 at 4096 vs
        # 39.3 at 2048), so this is scoped to the short rows.
        block = min(block, 2048)
    if wide <= 1024 or 4096 < wide <= 16384:
        cap = min(cap, 512)
    elif wide > 49152:
        # Above wide 16384 the capacity is not an occupancy knob any more, it
        # is what keeps the row out of the exact full-row rescan. Continuous
        # scores spread the threshold bin far wider than the integer-ranked
        # GLM ones: measured n_cand at B=64 is 215 at W=4096, 285 at 16384,
        # 541 at 32768 and 1090 at 65536. At 512 the last two rescan (16 % and
        # 72 % of MACA); 1024 covers 32768 (54 %) but not 65536 (57 %), where
        # 2048 reaches 264 %.
        cap = max(cap, 2048)
    else:
        # 16384 < wide <= 49152: the bin holds 541 candidates (median) and
        # `s_sel` is [next_pow2(group_topk) + cap] words, so 1024 is what puts
        # the CTA under 32 KB -- the ceiling for two CTAs per AP (histogram
        # 8 KB + s_sel 6 KB + the backend's [BLOCK] staging buffer 16 KB =
        # 30740 vs 34836 at 2048, and 32768 is the cliff). r24 tests whether
        # the second resident CTA is what the 16384 cells have and this one
        # does not.
        cap = min(cap, 1024)
    # Sampled single-read path: a 32768-wide row is 128 KB and one pass over it
    # is ~45 us of the ~125 us the kernel takes at B=256, yet the second pass
    # exists only to look the threshold up. Two sampled tiles give the
    # threshold to within one 11-bit digit, after which the one complete pass
    # can build the exact histogram and classify in the same sweep. The bin
    # around the threshold holds ~2048 elements at these widths (one digit
    # spans 2048 values at the top exponent), so the candidate capacity goes
    # to 2048 -- and at 29712 B of shared memory the CTA still fits twice per
    # AP, which is where the whole win lives (one CTA per AP measured 33 %
    # slower). The estimate never has to be canonical: it only has to bracket
    # the answer, and the guard after the pass checks that it did.
    samp = 0
    if 16384 < wide <= 49152:
        # The sampled single-read pass trades one extra sampled checkpoint for a
        # full row read, which only pays when there are enough rows in flight to
        # hide it.  Keyed on `wide` (next_pow2 of the row) it also caught rows
        # just above a power of two: L16385 (32 rows, 16385 columns) measured
        # 206 us with sampling against 32.1 us for its 16384-column neighbour,
        # a 6.4x cliff from one extra column.  Keep both the sampling and the
        # 2048-candidate capacity it needs for full grids only; a short grid
        # keeps the 1024 capacity, which is what buys the second CTA per AP.
        if batch > ONE_WAVE:
            cap = max(cap, 2048)
            samp = 2
    # Refine split: 10 bits in round 0, 11 in round 1.  A wider round 0 (11)
    # only helped while round 1 was expensive; with the `remain == 1` max
    # fast path (r49) firing is cheap, and the narrower round-0 scan wins on
    # every measured shape except one (B32 18.5 vs 19.1, pool2 18.6 vs 19.4,
    # L16384 29.8 vs 26.8 -- 11 of 12 shapes prefer 10 on log-geomean).
    r0bits = 10
    # The exact 16K one-wave shape is the one case where the wider first
    # digit wins decisively: it removes the expensive round-1 scan for the
    # worst row without imposing the extra 2048-bin cost on ragged 16K rows
    # or on larger batches.
    if int(score.shape[1]) == 16384 and batch <= ONE_WAVE:
        r0bits = 11
    r0s = shift0 - r0bits
    r0n = 1 << r0bits
    r1n = 1 << r0s
    histn = max(bins, r0n)

    if wide > 4096:
        # With four CTAs per resident AP, a deeper pipeline buys nothing and
        # costs registers: prefill_63ki 202.0 us at 1-2 stages vs 216.9 at 3.
        # The 1024-lane rows have a single-tile loop and are flat across
        # depths (15.2 us at 1 or 3), so they keep the default.
        num_stages = min(num_stages, 2)
    SMEM_BUDGET = 65536 - block * 4

    def _fits(c):
        # The two staging lists hold element indices at two bytes each, so the
        # candidate capacity costs c * 2 and the selected set next_pow2(group_topk)
        # * 2. The digit array (when the wrapper decides to keep one) is the
        # wide*2 term.
        return (bins * 4 + 16 + c * 2 + _next_pow2(group_topk) * 2
                + wide * 2) <= SMEM_BUDGET

    # The digit array costs 32 KB at the wide shapes, which drops the CTA to
    # one per AP. That is free when the grid is one wave (no other CTA to hide
    # latency behind) and a big loss when it is not: prefill_63ki 230.8 ->
    # 313.8 us at B=1024, while decode_63ki 46.8 -> 42.3 us at B=2 from
    # dropping the second row read.
    # r42 tried a power-of-two window smaller than the row (to give 16385-column
    # rows a partial digit) and regressed every digit shape by ~2.5 us: the
    # per-tile runtime branch duplicates the staging loop body, and the extra
    # registers/issue cost more than the re-read it saves.  Full coverage only.
    dig = 0
    if r0bits < 12 and 1024 < wide <= 16384 and batch <= ONE_WAVE:
        for c in (cap, 1024, 512):
            if c <= cap and _fits(c):
                cap, dig = c, 1
                break
    table_width = page_table.stride(0) if page_table is not None else 0
    po = page_table if page_table is not None else score
    oo = topk_indices_offset if topk_indices_offset is not None else lengths
    so = row_starts if row_starts is not None else lengths
    qo = seq_lens if seq_lens is not None else lengths
    to = page_table_row_index if page_table_row_index is not None else lengths

    # Wide one-wave rows: the row is much larger than one 512-thread CTA can
    # stream and the grid cannot fill the 104 APs, so split the row across
    # NSEG CTAs in four launches (see the split kernels above).  Fewer than
    # ~16K columns does not pay: the per-CTA histogram fixed cost is ~4.5 us
    # and the split has to beat it (r40 seg probe).
    if batch <= ONE_WAVE and wide >= 16384:
        # Ragged 16K boundaries are dominated by per-segment histogram and
        # classify fixed costs.  NSEG=2 gives four 4096-wide tiles for
        # 16125..16384; the one-column-over-power-of-two case W=16385 needs
        # NSEG=3 (six tiles) to avoid the extra predicated tile.  Overflow
        # rescan remains unchanged.
        if 16125 <= int(score.shape[1]) <= 16384:
            nseg = 2
        elif int(score.shape[1]) == 16385:
            nseg = 3
        else:
            nseg = 16 if batch <= 4 else (4 if wide <= 49152 else 8)
        # One segment should be about one tile: a wider tile leaves half its
        # lanes predicated off, which is issued work on this AP (D16).
        _chunk = (int(score.shape[1]) + nseg - 1) // nseg
        sblock = min(4096, max(512, 1 << (_chunk.bit_length() - 1)))
        _split_topk(score, lengths, output, pool_size, page_table,
                    topk_indices_offset, row_starts, seq_lens,
                    page_table_row_index, width, out_cols, table_width,
                    token_topk, group_topk, nseg, sblock, cap,
                    _next_pow2(group_topk), r0s, r0n, r1n)
        return

    kargs = (score, lengths, output, po, oo, so, qo, to, width, out_cols,
             table_width, token_topk, group_topk)
    # Everything the launch depends on except the eight pointers (see the
    # fast-launch comment above `_direct_launch`).
    lkey = (score.dtype, lengths.dtype, output.dtype, batch, score.shape[1],
            width, out_cols, table_width, token_topk, group_topk, pool_size,
            page_table is not None, topk_indices_offset is not None,
            seq_lens is not None, row_starts is not None,
            page_table_row_index is not None, block, bins, shift0, cap,
            num_warps, num_stages, samp)
    if _LAUNCH_CACHE and _direct_launch(lkey, kargs[:8], batch):
        return

    kkw = dict(
        POOL=pool_size,
        POOL_PAD=_next_pow2(pool_size),
        HAS_PT=page_table is not None,
        HAS_OFF=topk_indices_offset is not None,
        HAS_TAIL=seq_lens is not None,
        HAS_STARTS=row_starts is not None,
        HAS_TROW=page_table_row_index is not None,
        BINS=bins,
        LOG2_BINS=bins.bit_length() - 1,
        SHIFT0=shift0,
        HISTN=histn,
        R0S=r0s,
        R0N=r0n,
        R1N=r1n,
        BLOCK=block,
        CAP=cap,
        CAPP=_next_pow2(group_topk),
        # Register-resident single-read path only for short rows: at 16384
        # lanes the row costs 32 words/thread, and keeping it live across the
        # histogram + cumsum select costs more than the second read saves
        # (prefill_63ki 230.7 -> 273.3 us, decode_63ki 46.8 -> 69.4), while at
        # <=1024 lanes it is a clean win (prefill_3ki 31.2 -> 26.8).
        WIDE=wide if wide <= 1024 else 0,
        # Single-read staging for the tiled path: one 11-bit digit per element
        # (2 bytes each) in shared memory. Worth it only for long rows, and
        # only when it fits next to the histogram and the staging arrays --
        # the backend needs ~8 KB beyond the sum of the allocations, and the
        # AP has 64 KB.
        DIG=dig,
        DIGW=wide if dig else 1,
        SAMP=samp,
        # Drop-one fast path: only reachable when a row can hold exactly one
        # group more than it selects, i.e. width <= group_topk + 1.  Gated on
        # the visible width so the extra reductions are not even compiled for
        # the wide bands (they would only add register pressure there).
        DROP1=(1 if (int(score.shape[1]) <= group_topk + 1
                    and seq_lens is None) else 0),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    kern = kpool_topk_transform_kernel[(batch,)](*kargs, **kkw)
    _cache_launch(lkey, kern, kargs, kkw, batch)
    _cache_fast_launch(fkey, kern, kargs, kkw)


# 算子名（提交要求）
dsa_kpool_topk_transform = fast_kpool_topk_transform_fused

