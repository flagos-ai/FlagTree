# What `tle.gpu` can and cannot reduce on KL3

Each limit has the repro that established it. Everything here was measured on
`arch 3` with `XPU_EVENT_KL3_ENABLE=1` on an idle card.

## Reduction axis

`triton_xpu.reduce` only anchors on the **last** axis. `TLECoreTiling.cpp:383`:

```cpp
if (reduceOp.getAxis() != shape.size() - 1) return reject(...);
```

Related: `Utility.cpp:121` (`emitOffsetForSliceLayoutXPU`) and `ReduceOpToLLVM.cpp:190`
(`if (op.getAxis() == 1)`). So `tl.sum(x, axis=0)` on a core-tiled tile is not
available, and a middle-axis reduce has to keep a per-column accumulator and fold the
reduced axis across loop iterations (elementwise adds only). This is the single
constraint that forces two passes; one pass would need either the axis-0 reduce or a
transposing GM -> LM copy, which `tle.gpu.copy` cannot express.

## Masking a tile that came through LM

| spelling | result |
|---|---|
| `tl.where(coff + col_ids < N, aval, 0.0)` inside the reduction | bf16 silently wrong on (600, 40999); f16 fails to compile at every tile size ("TLE kernel stack is over the local-memory budget", pipeline pass off as well) |
| `tl.where(...)` elementwise, not in a reduction | still wrong: (3, 33, 512) f32 off by 398, (3, 32, 1000) off by 13.6, (3, 33, 1000) returned nan / 2e38 |
| `tl.load(local_ptr, mask=..., other=0)` | does not compile: `'tt.load' op failed to verify that mask type matches ptr type` — the pointer is `!tt.ptr<T, 0>` from `tle_local_ptr` and the verifier has no mask rule for that address space |
| descriptor `padding="zero"` | not honoured on this path |

**Use zero-fill instead**, and only where the overhang leaves the tensor. See SKILL.md;
the short version is that a clamped `tle.gpu.copy` leaves stale LM bytes, and clearing
the buffer before the one short step makes an arbitrary extent exact (verified on
f32/f16/bf16/int8/16/32/64 over 64x100, 64x513, 256x1000, 8192x8191, 600x40999 at
YBLOCK 128, 64x3).

Where the overhang stays *inside* the tensor — a `(B * N, K)` descriptor, where a K
block runs into the next row and an N block into the next batch — zero-fill cannot
help: (3, 32, 100) f32 off by 6.6, (3, 512, 1000) by ~45. Shift the last block back on
the contiguous axis (`k_off = min(kb * KBLOCK, K - KBLOCK)`, harmless because the
overlap recomputes the same columns); keep a divisor on the reduction axis.

## On the GM-pointer side

Masking works, but **do not clamp the offsets to the mask** to keep addresses in range:

```python
off_c = tl.where(keep, off, 0)
a = tl.load(ptr + off_c, mask=keep, other=0)     # mask stops applying
```

The dropped lanes then read element 0 and contribute — measured on (3, 1000, 129) f32
as exactly the padding lanes' worth of error (24 lanes, diff 24.3). The overrun that
clamping was meant to fix is real (a power-of-two tile addresses past the last row and
faults at 1-2 bytes per element: (600, 40999) int8 raises an illegal access), so the
answer is to keep narrow dtypes off such a kernel, not to clamp.

## dtypes

| dtype | status |
|---|---|
| f32, f16, bf16 | exact |
| int8/16/32/64 | exact **with an int64 accumulator** (aten's rule); f32 accumulation starts losing counts at 2^24 |
| uint8 | exact with an int64 accumulator (checked on 6 shapes) |
| bool | no LM dtype; cast to int8 on the host (values are 0/1, so it is exact) |
| float64 | **does not exist on the device** — `torch.randn(dtype=torch.float64, device="cuda").dtype` is `torch.float32`, `element_size()` 4, 1e300 -> inf. A kernel declaring `tl.float64` reads f32 bytes as doubles: an all-ones (64, 64) reduce returns 1.0 / 1.75 alternating instead of 64.0. Not a compiler bug, and no f64 path is needed. |

## Descriptors

- `TensorDescriptor.from_tensor` requires **last stride == 1**. An extent-1 last
  dimension breaks this: torch is free to give it any stride and hands back `M`, so
  `inp.view(M, 1)` is rejected outright. Handle `N == 1` on the host — it is a copy.
- A **block larger than the extent is legal** and clamps. This is what lets XBLOCK
  exceed M, and what lets a `[1, n]` reduce finish a small flat residue.
- The block shape and the row pitch are separate: pass the parent row stride when
  reducing a **column slice**, otherwise the DMA strides wrong and silently reduces the
  wrong columns (a hardcoded `N` gave wrong results on (64, 100)).
- Flat-launcher operand ABI for a non-sdnn descriptor: base pointer, then `.shape`
  (i32), then `.strides` (i64).

## `tritonxpu-tle-pipeline`

Requirements, all of them:

- `TRITONXPU_TLE_PIPELINE=1` at pass-construction time — the first compile, so setting
  it at import is early enough.
- `tl.range(..., num_stages>=2)` on the loop; the pass is a no-op otherwise.
- The rotated buffer must be a loop-external `local_alloc`, **read-only inside the
  loop**, addressed by the loop variable plus invariants. A zero-filling `tl.store`
  inside the body disqualifies it — peel that step out of the loop.
- At least **64 bytes per core** in the rotated buffer. Below that, lowering fails with
  `invalid element type in packLLElements. Expected '!llvm.ptr<2>' but got '!llvm.ptr'`.
  Swept over NBLOCK x KBLOCK x dtype, the boundary is exactly 64 B/core, and every
  failing tile lowers fine at `num_stages=1` — so gate the depth, not the shape.
- `fitBudget` counts only `local_alloc` and runs even with the pass off, which is why
  the tile has to come down when you enable it.

## Compiler bug to route around

A runtime-bounded `range` loop nested inside a `tl.static_range` unroll miscompiles.
A per-row kernel of this shape:

```python
for ri in tl.static_range(BLOCK_M):
    ...
    for c in range(0, NW, TL):      # NW is a runtime argument
        acc += tl.load(base + c + off)
```

returned the right answer only for `ri == 0` — 75 of 600 rows, exactly one per unrolled
group of 8. Use a 2-D tile, or one program per row.
