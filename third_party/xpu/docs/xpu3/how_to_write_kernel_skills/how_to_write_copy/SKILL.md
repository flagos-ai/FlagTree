---
name: xpu-tle-dsa-copy
description: Move data-movement operators (copy_, permute_copy, unfold_copy, as_strided_copy, cat, index_copy, view_copy...) onto tle.dsa/SDNN on Kunlunxin KL3, and debug them when they are slow or silently wrong. Use this skill whenever someone asks why a non-contiguous / strided / transposing / broadcasting copy is slow on XPU, wants to know whether tle.dsa can express a layout, sees a copy-like Triton kernel produce wrong numbers or zeros on the SDNN path, or is benchmarking these operators and getting numbers that look impossible (all latencies equal, sudden 100x outliers). Also use it before hand-tuning tile sizes or block counts for such a kernel, because on this hardware the shape of the transfer matters far more than the tuning constants.
---

# Data movement on KL3: match the DMA engine, not the element

A copy operator on KL3 lives or dies by one question: **does every global-memory
access it performs correspond to a shape the DMA engine actually has?** Tuning
constants cannot rescue the wrong shape — a per-element gather of a 4096x4096 f32
window measured 32.8ms where a 2D row transfer of the same bytes takes 114us and
`aten::copy_` takes 157us. So the work is: find the hardware's native shapes, map
the layout onto them, and refuse (fall back) when it does not map.

This skill is the accumulated result of doing that for `copy_`, `alias_copy`,
`permute_copy` and `unfold_copy`. It is written for KL3 (`arch 3`); the SDNN
details differ on KL4/KL5.

## Start by reading the hand-written kernel

Before designing anything, look at how XDNN does the same move. It is the ground
truth for what the hardware supports, and it is readable:

- `baidu/xpu/api/src/wrapper_aten/<op>.cpp` — the arch dispatch. Find the
  `xpu3_wrapper` (not `xpu1_wrapper`/`xpu2_wrapper`, which target older parts) and
  read the conditions: they tell you which shapes get a dedicated kernel and which
  get decomposed.
- `baidu/xpu/api/src/kernel/kunlun3cpp/kunlun3cpp_aten/*.xpu` — the kernels.

Two that matter for any copy:

- `memcpy_2d_sdnn.xpu`: `dma_cfg_2d(loop, dst_stride, src_stride)` — **`loop` rows
  of a contiguous run, with independent src and dst row strides**. This is the
  native shape. One DMA descriptor per row, not per element.
- `transpose_021_sdnn_bsp.xpu`: a 3-core BSP pipeline — `dmai_2d` in, then
  `ds_shuffle_coa_1d` **on chip**, then `dmao_2d` out, over ping-ponged uni_sram
  halves. The lesson is stronger than the code: native never reads GM with an
  element stride. Transposition happens on chip; both GM sides stay row-contiguous.

If the operator you are porting has no dedicated native kernel, check whether the
wrapper decomposes it into a sequence of 2D/3D steps with a GM scratch buffer
(`transpose.cpp`'s execution plan does exactly this). That decomposition is also
available to you.

## Map the layout, then pick a path

Collapse the copy first: sort dimensions by `|dst stride|`, merge neighbours where
both stride sets are consistent (`ss == pss * pn and ds == pds * pn`), and keep the
innermost two for the tile while the rest become a scalar base offset off the
program id. Outer dimensions cost nothing in the transfer, so put as many there as
the grid can carry.

Then, with `cols`/`s_col` the innermost extent and its src stride, and
`rows`/`s_row` the next one out:

| condition | path |
|---|---|
| both sides contiguous, same dtype | flat TMA tile on the cluster (`tle.gpu`), bind the launch once |
| `s_col == 1` | 2D row transfer: rows of a contiguous run, row strides independent |
| `s_col != 1` and `s_row == 1` | read a square tile, transpose it on chip (`tl.trans`), write it out |
| anything else | return False, let the caller's pointwise kernel do it |

Falling back is a feature, not a defeat: the pointwise kernel addresses the
destination element by element and has none of these restrictions. Your job is to
take the layouts where the DMA shape applies and leave the rest alone.

## The refusals that matter

Three layouts are **silently wrong** if you attempt them on this path — no error, no
fault, just different numbers. They have to be rejected on the host side:

1. **a strided destination run** (`y[::2]`): the stride is dropped and the data goes
   out packed.
2. **an innermost run with src stride 0** (broadcast from a single element): it comes
   back as consecutive elements.
3. **int -> float casts**: all zeros.

Number 3 is not a `tle.dsa` artifact — a plain `tl.load(...).to(tl.float32)` in a
kernel launched with `is_sdnn=True` does the same, so it is the SDNN cast. Native
casts on the cluster (`cluster_cast_kl3`, SIMD) instead, which is why aten gets this
right and you cannot lift its approach into an SDNN kernel.

The dtype limits are the ones to watch (i32
faults, i64 is rejected at compile time, bf16 cannot be a buffer element type) and
the trick that sidesteps most of them: a pure move only cares about width, so view
the bits as 1/2/4-byte elements, and split an 8-byte element into two 4-byte ones
with `view(torch.float32)`.

Whenever you add a path, write the matching refusal in the same change. A silent
wrong answer costs far more than a fallback.

## Tails go through `sizes`, never through a mask

```python
buf = tle.dsa.alloc([ROWS, COLS], DTYPE, tle.dsa.UNI_SRAM)
tle.dsa.copy(src_ptrs, buf, sizes=[row_tail, col_tail])   # in: narrows the DMA
tle.dsa.copy(buf, dst_ptrs, sizes=[row_tail, col_tail])   # out: becomes a store mask
```

A masked load hides its extent in a view of a staging buffer, and the dsa rewrite
that retargets the DMA cannot carry that over — the compiler rejects it rather than
moving the wrong amount. Keep the buffer statically full-width so the SRAM allocator
can place it, and shrink only the transfer.

## Tile size: measure, do not reason

On-chip tile sizes on this part do not behave smoothly. For the transposing path,
64x64 works and its neighbours do not: on a 2048x2048 f16 transpose, 64x64 measured
66us while 128x128 and 32x256 both landed around 120-140ms. Treat any tile constant
as an empirical result with the measurement written next to it, and re-measure when
the dtype or the path changes.

## `do_not_specialize`: "first ≈ median" means every call reloads

Copy operators allocate a fresh output per call, and the XPU caching allocator hands
back pointers from varying divisibility classes. If pointers and scalars are on the
specialization key, each new class sends the launch back for another kernel — with a
warm cache that is a 10-30ms load, not a 1-4s compile, so it does not look like
compilation at all.

The diagnostic is the shape of the per-call timings:

- **first >> median** — one compile, then steady state. Normal.
- **first ≈ median, both large** — every call is loading a kernel. Fix the key.

Measured over one sweep of six strided copies, same code: 482s of wall clock with
specialization on, 12s with `do_not_specialize` covering pointers and every runtime
scalar. Two shapes that looked like pathological kernels (14.7ms and 25.0ms per call)
were reloads, and became 334us and 199us.

Nothing in a copy kernel's runtime arguments — pointers, extents, strides — changes
the generated code, so there is nothing to lose. Keep only the real constexprs (rank,
tile size, dtypes, mode flags) on the key.

## Do not gate on size without checking what the fallback is

A size threshold ("a small copy is not worth an SDNN launch") is only as good as the
thing it falls back to. When the fallback changed from aten to a Triton pointwise
kernel, the measured crossover moved from 32MB to about 1MB, and the remaining
deficit below that was 3-8% of a host-bound path — not worth a constant, an
environment override and a per-caller argument. Prefer one rule for every caller and
delete the knob.

If a caller does need the path unconditionally, the reason is usually that its
fallback is *wrong* rather than slow — `unfold_copy`'s generic kernel drops 12 of 72
elements on overlapping windows. Write that reason at the call site.

## Measurement discipline

This is where the time goes if you are careless. `references/measurement.md` has the
details; the short version:

- **Pick an idle card explicitly.** Card 0 is disabled on this machine; a busy
  neighbour distorts everything. One task per card.
- **Use a fixed compile cache while investigating** (`FLAGGEMS_CACHE_DIR=/tmp/...`).
  A fresh cache per run buries small shapes under compiles. Save fresh-cache runs for
  final acceptance.
- **Read both absolute latencies before believing a ratio.** A 2.90x "win" turned out
  to be a torch baseline that had drifted from 0.082ms to 0.540ms while our side was
  unchanged.
- **Discard runs whose latencies collapse onto two constants** (every case 6.44ms or
  12.88ms, exactly 2x apart, on both sides). The timer has degenerated and the run
  says nothing about the code. Real numbers for small copies are single-digit
  microseconds.
- **A/B inside one process** where you can; cross-process drift on this machine
  reaches 35%.

`scripts/sweep_copy.py` is a template for the two things worth automating: a
correctness sweep over layouts and dtypes (windows on either side, tails in both tile
dimensions, broadcast rows, rank 3-5, transposes, every dtype, conversions, bool), and
a three-way timing comparison of your path against the fallback and native.

## Reporting

Say which path each shape took, and give absolute latencies for your path, the
fallback and native side by side — a ratio alone hides both baseline drift and the
case where everything is slow. When a shape falls back, say so explicitly instead of
letting it disappear into an average: a fallback is a design decision, a mystery is
not.

## Reference files

- `references/measurement.md` — card selection, cache handling, and the four
  measurement artifacts that produced wrong conclusions in this work
- `scripts/sweep_copy.py` — correctness sweep and three-way timing template
- `copy_family_case_study.md` (in `third_party/xpu/docs/xpu3/how_to_write_kernel_skills/`, not shipped
  with the installed skill) — the change this came from, with before/after numbers and
  the dead ends

A worked implementation of all three paths is
`FlagGems/src/flag_gems/runtime/backend/_kunlunxin/utils/tle_copy.py`: `_collapse` for
the folding, `_tle_dsa_row_copy_kernel` and `_tle_dsa_trans_copy_kernel` for the two
SDNN paths, and the host-side dispatch in `tle_copy()` for the refusals. Read it before
writing a new one — the comments carry the measurements.
