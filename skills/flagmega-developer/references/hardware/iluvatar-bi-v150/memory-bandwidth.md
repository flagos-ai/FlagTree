# Iluvatar BI-V150 (COREX) memory bandwidth characteristics

Target-specific memory behavior that inverts the vectorization guidance most
CUDA experience teaches. Read this before tuning any bandwidth-bound kernel on
this backend, and before attributing a read-bound kernel's shortfall to the
HBM itself.

## Avoid vectorized loads; wide loads lose ~45% of read bandwidth

**Vendor-confirmed (Iluvatar, 2026-10-08): avoid vectorized loads on BI-V150.**

**Theoretical peak is 819.2 GB/s per die** (2048-bit bus at 1.6 GHz DDR, from
`cudaDeviceProp.memoryBusWidth` / `memoryClockRate`). BI-V150 is a dual-die
board (`ixsmi -q` reports `MultiGPU Board: Yes` and `GPU Position: 0|1`), so
one CUDA device is one die: 16 MPs at 1.5 GHz, 32 GB, 16 MB L2, cc 7.1, and
**`warpSize` 64**.

Measured achievable bandwidth, as a fraction of that per-die peak:

| path | GB/s | % peak | per-MP rate |
|---|---|---|---|
| store, any width | 749 | 91% | 31 B/cyc |
| SME G2S, double-buffered | 730 | 89% | — |
| SME G2S, naive (drain every iteration) | 690 | 84% | — |
| **32-bit load** | **676** | **82%** | 6.9 ld/cyc |
| 64-bit load | 589 | 72% | 3.1 ld/cyc |
| **128-bit load (`float4`)** | **373** | **46%** | 0.97 ld/cyc |
| 16-bit load | 574 | 70% | 12.0 ld/cyc |

**Root cause: load-instruction issue rate, not HBM and not latency.** Wide
loads cap at roughly one load instruction per cycle per MP, so a 128-bit load
moves only 16 B/cycle/MP while scalar loads sustain ~28 B/cycle/MP. Stores are
unaffected at every width. Three independent observations localize the cap to
the SM side rather than the memory system: read bandwidth is flat from 32K to
2M resident threads (a latency bound would improve with occupancy), dependent
load latency is 122 ns, and read bandwidth scales linearly from 1 to 16 MPs
(28 → 390 GB/s with 128-bit loads) and then stops dead.

**How to apply.**

- Opt out with the `disable_load_vectorize` compile option, set like any other
  Corex option — as a launch kwarg, or via `options=` on `triton.compile`:

  ```python
  kernel[grid](x, out, BLOCK_SIZE=1024, disable_load_vectorize=True)
  ```

  It becomes the `disable-load-vectorize` option on
  `convert-triton-iluvatargpu-to-llvm`, so only lower-to-LLVM acts on it: the
  TTIR/TTGIR handed to the pass is byte-identical either way, and contiguity
  analysis, every earlier pass, stores, atomics and SME async copies are all
  unaffected. Measured on a read-bound reduction at 64 CTAs/MP: 342 GB/s
  vectorized (4-wide loads) vs 455 GB/s scalar, a 1.33x gain.
- Prefer scalar/narrow loads in bandwidth-bound kernels on this target. Do not
  widen loads to `float4`/128-bit for throughput; that choice costs 45% of read
  bandwidth here even though it is the right default on NVIDIA.
- Keep stores wide if convenient — the store path reaches 91% of peak at every
  width and is not the bottleneck.
- In Triton this surfaces through the block shape: a pure streaming read at
  `BLOCK=256` measured 654 GB/s, while `BLOCK >= 1024` (which Triton
  vectorizes) dropped to ~450 GB/s. When a read-bound Triton kernel
  underperforms, test a narrower `BLOCK` before assuming a memory-system limit.
- Extra independent address streams per thread do not help once the load width
  is right (1 stream beat 2 and 4 at every launch shape with 32-bit loads).
- Coalescing still dominates everything else: assigning each thread its own
  contiguous region collapses to 40-64 GB/s, about 7x worse than a grid-stride
  pattern. Never give threads contiguous per-thread ranges.
- Working sets below ~16 MB measure L2, not HBM; bandwidth saturates by
  ~128 MiB.
- Mixed kernels written with `float4` inherit the read limit: copy 453 GB/s
  (2x traffic), triad 405 GB/s (3x), `cudaMemcpy` D2D 589 GB/s. PyTorch matches
  the 128-bit figures (`sum` 344, `fill_` 635, 3-buffer triad 581) because it
  also vectorizes loads.

## The backend's 128-bit cap is tuned for one CTA per SM

`getVectorSize()` in
`third_party/iluvatar/backend/lib/TritonILUVATARGPUToLLVM/Utility.cpp` raises
the global-load vector width to 128 bits for this target, with a comment citing
"C++ float4 microbenchmark, perf-iteration/ITERATION.md Trial 74: 366 GB/s vs
167 GB/s at 16 CTAs". **That measurement is reproducible and its conclusion is
correct at 16 CTAs, but the ordering inverts at higher occupancy.** Measured
read bandwidth per load width against CTA count (block = 1024 threads, 512 MiB
buffer):

| CTAs | CTAs/MP | 32-bit | 64-bit | 128-bit | winner |
|---|---|---|---|---|---|
| 16 | 1.0 | 169 | 324 | **353** | 128-bit |
| 32 | 2.0 | 320 | **541** | 366 | 64-bit |
| 64 | 4.0 | **596** | 588 | 382 | 32-bit, 1.56x |
| 128 | 8.0 | **658** | 569 | 366 | 32-bit, 1.80x |
| 256 | 16.0 | **660** | 574 | 367 | 32-bit, 1.80x |
| 1024 | 64.0 | **664** | 591 | 383 | 32-bit, 1.73x |
| 2048 | 128.0 | **676** | 604 | 389 | 32-bit, 1.74x |

At one CTA per SM there is too little memory-level parallelism to saturate
anything, so moving more bytes per issued instruction wins and 128-bit is the
right choice. Past ~4 CTAs/MP the binding constraint becomes the ~1 load
instruction/cycle/MP issue limit, and wide loads plateau near 373 GB/s while
scalar loads keep climbing to ~676.

**This matters because the two regimes correspond to two different kinds of
FlagMega kernel on this target.** A megakernel using in-kernel grid barriers
(`tle.distributed_barrier`) is structurally pinned to one CTA per SM: this
backend reports `cooperative_launch_admission=False`, so
`_validate_resources` in `python/triton/flagmega/runtime/prepared.py` rejects
any grid larger than `available_sm_count` to avoid a barrier deadlock — 16 CTAs
on a 16-MP die. Those kernels sit exactly at Trial 74's operating point and the
128-bit cap is right for them. Ordinary (non-grid-synchronized) kernels launch
many CTAs per SM, land in the inverted regime, and are the ones that lose up to
1.8x read bandwidth to the same cap.

`getVectorSize()` takes only the pointer and axis info; it has no occupancy or
CTA-count input, so it cannot distinguish these cases automatically. When
tuning a read-bound kernel that is **not** grid-barrier-constrained, do not
assume the backend's default width is right for it — set
`disable_load_vectorize=True` and measure. The default is deliberately
unchanged, since narrowing it would regress the megakernel case the 128-bit cap
was added for.

Stores are insensitive to width at every CTA count (32-bit vs 128-bit within
0.97-1.04x from 16 to 1024 CTAs), so this concerns loads only and wide stores
need no special handling.

## Getting past 82%: the SME G2S path, double-buffered

For read-bound kernels, the SME global-to-shared engine is the only measured
way past the scalar-load ceiling, because it does not return data through the
LSU/register path. It reaches 89% of peak, close to the store path's 91% — but
only when double-buffered.

Issue the next stage's tiles **before** `sl_waitcnt` drains the current stage.
Naive issue-wait-consume gives 84%; double-buffering gives 89%. Counter to
intuition, more tiles in flight per warp made it worse (730 → 698 GB/s going
from 1 to 4 tiles per warp), so prefer one tile per warp with a moderate block
size.

ABI, per `emitIluvatarSmeTileLoads` in
`third_party/iluvatar/include/triton/Conversion/TritonGPUToLLVM/Utility.h`:

```c
__ivcorex_sme_load_16x1b64(smem_addr_i32, abase_i32x4, gmem_byte_off, 0);
// abase = { gptr_lo, gptr_hi, -1, row_stride_bytes }
// row stride must be a positive multiple of 64 bytes
// one 16-row x 64-byte (1 KB) tile per warp per instruction
__ivcorex_sl_waitcnt(0);  // drain outstanding G2S transactions
```

`abase` must be a native `ext_vector_type(4)` int, **not** CUDA's `int4`
struct — passing `int4` fails to compile.

**ivcore11 has SME G2S but not plain `cp_async`.** The `__ivcorex_cp_async_*`
builtins report `needs target feature async-copy` on ivcore11 (they are
available on ivcore10/ivcore20), and forcing
`-Xclang -target-feature -Xclang +async-copy` crashes codegen. Use the SME
builtins on this target.

Always confirm SME actually fired by grepping the device assembly for
`sme_load` before trusting a measurement:

```sh
clang++ -std=c++17 -O3 --cuda-path=/usr/local/corex \
  --cuda-gpu-arch=ivcore11 -D__MR__ --cuda-device-only -S kernel.cu -o kernel.s
grep -c sme_load kernel.s
```

The same check distinguishes real load widths: `ml_lsa_load_a64_dwordx4` is
128-bit, `dwordx2` is 64-bit, `dword` is 32-bit. High per-thread unroll or
stream counts silently spill wide loads down to `dwordx2`, which will confound
any conclusion drawn without reading the assembly.

## Measurement notes

Establishing the numbers above required discarding several self-inflicted
measurement errors; repeat benchmarks here with these in mind.

- Do not put an `atomicAdd` on a single address inside a timed kernel as a
  verification counter — the serialized atomics dominated the timing and
  flattened every result to ~370 GB/s. Verify in a separate untimed kernel.
- Derive byte counts from the exact loop geometry. A `chunk = n/max(R,W)`
  helper over-reported read bandwidth as 574 GB/s because the loop covered only
  part of the buffer.
- A read loop with a single accumulator measures load latency, not bandwidth.
- `torch.add(a, a, out=c)` is 2x traffic, not 3x; use distinct buffers for a
  real triad.
- Confirm a fast store figure is genuine traffic. The 749 GB/s store result was
  validated three ways: a position-dependent pattern read back on the host,
  per-iteration-varying data (743.7 vs 742.6 GB/s, ruling out repeat-store
  elision), and incompressible hashed bits (735 GB/s, ruling out memory
  compression).
- Benchmark stdout redirected to a file is block-buffered; use `stdbuf -o0` or
  stderr, or a hang looks like a silent failure with no output at all.
