# Low-precision arithmetic on HCU

This implementation is validated on `hip/gfx936` with 64-thread warps. Other
HCU targets retain target-dependent instruction selection; their hardware
results are not established by the gfx936 tests.

| Format | Storage | Matrix implementation on gfx936 |
| --- | --- | --- |
| FP8 E4M3FN | `tl.float8e4nv` / `torch.float8_e4m3fn` | GPU software decode, native FP16 MMAC, FP32 accumulation |
| FP4 E2M1 | Two codes per `uint8`, first value in the low nibble | GPU software decode, native BF16/FP16 MMAC, FP32 accumulation |
| INT32 | Signed `tl.int32` | Accumulator/output for integer dot; not an A/B operand |
| INT8 | Signed `tl.int8` | Native INT8 MMAC for applicable tiles; integer vector/scalar paths otherwise |
| INT4 | Signed two's-complement nibbles in `uint8`, low nibble first | GPU unpack to INT8, then the INT8 dot implementation |

FP4 and INT4 have explicit packed representations, not independent pointer
dtypes. Raw copies retain every byte, including an unused high nibble. The
algorithm must mask unused logical elements before matrix computation.

## Conversions

Ordinary `.to()` supports E4M3FN/E5M2 conversions with FP16, BF16, FP32 and
FP64. Downcasts support default/explicit RTNE and RTZ with finite saturation,
signed-zero preservation and NaN-class preservation; NaN canonicalization is
allowed. FP64 uses a round-to-odd FP32 intermediate before FP8 rounding.
Integer/FP8 conversion uses an exact FP32 bridge over the finite FP8 integer
range. Floating-to-integer conversion retains Triton's RTZ and representable
input requirements.

INT32-to-BF16 constructs the rounded bits directly, avoiding FP32 double
rounding. For example, `16842753` produces BF16 `0x4b81` (`16908288`).

The HCU backend exposes these packed conversions through the existing
`tl.ext` mechanism:

| API | Contract |
| --- | --- |
| `tl.ext.fp_to_fp4(x, axis=-1, fp_downcast_rounding="rtne")` | FP16/BF16/FP32/FP64 to packed E2M1; even packing axis; RTNE/RTZ; overflow, Inf and NaN saturate to signed magnitude 6; no implicit scale |
| `tl.ext.fp4_to_fp_scaled(x, scale, elem_type, axis=-1)` | Decode packed E2M1, apply E8M0 scale in FP32, narrow to FP16/BF16 with RTNE; scale may be None, expanded per pair, or grouped per 32 logical values |
| `tl.ext.unpack_int4(x, axis=-1)` | Packed uint8 to signed INT8 in [-8,7] |
| `tl.ext.pack_int4(x, axis=-1)` | INT8/INT32 to packed bytes, modulo 16, without saturation or scaling |

For floating-to-INT4 modulo packing, explicitly convert finite representable
values to INT32 with `.to(tl.int32)`, then call `pack_int4`.

## Dot and scales

Ordinary signed INT8 dot, including decoded INT4 inputs, uses INT32
accumulation and output with modulo 2**32 arithmetic. Explicit accumulators
are checked against the actual result dtype. INT32 is not supported as an
A/B operand of dot or scaled dot; it remains available for storage,
conversions, data movement, and integer accumulation/output.

Use `tl.dot_scaled` with the `"e2m1"` tag for packed FP4. Either scale may be
None; both None express an unscaled FP4 dot. E8M0 scales are uint8, with one
code per 32 logical K values and shapes `[M,K/32]` and `[N,K/32]`. Code 0 is
`2**-127`; code 255 produces NaN. Scaled tiles require K divisible by 32.
Masked tiled loops can handle a partial final logical group.

Scaled software decomposition decodes operands to BF16, or FP16 when either
input has the `"fp16"` tag. Each unscaled 32-element dot accumulates in FP32;
the combined scale exponent is applied with GPU `ldexp`, then the partial
result is added to the FP32 accumulator. This avoids materializing scaled
BF16 subnormal operands or overflowing each scale separately. The underlying
MMA dtype's behavior still applies to original BF16/FP16 operands.

HCU accepts `"int4"` and `"int8"` in `tl.dot_scaled`. With FP32 output,
integer inputs use **FP32 scales** with shapes `[M,K/32]` and `[N,K/32]`.
K is divisible by 32 and INT4 inputs are packed along K. Each group's
integer result wraps modulo 2**32, converts to FP32, and is multiplied by
the left and right scales before being added to the FP32 accumulator.
None means unit scale. FP32 arithmetic follows `enable_fp_fusion`.

With both scales set to None, `out_dtype=tl.int32` selects unscaled integer
dot. Packed INT4 supports K or M/N packing through `lhs_k_pack` and
`rhs_k_pack` in this mode.

```python
# K-packed INT4 and signed INT8 with FP32 group scales.
scaled = tl.dot_scaled(a_packed, a_scale, "int4", b_int8, b_scale, "int8")
# Packed INT4 dot with an INT32 accumulator and output.
integer = tl.dot_scaled(a_packed, None, "int4", b_packed, None, "int4",
                        acc=acc_i32, out_dtype=tl.int32)
```

Compile-time Python `UserWarning` identifies software matrix paths. Native
classification follows actual HCU lowering, rather than an architecture
table copied from an uncompiled AMD implementation. Smaller tiles can use
vector dot or scalar arithmetic; `native=false` describes the matrix
strategy, not CPU execution.

## Data movement and TLE async loads

Ordinary load/store and TLE shared-memory operations preserve raw bits for
these storage types. `tle.load(..., is_async=True)` can use native
buffer-to-LDS instructions, with waits and CTA barriers, when pointer range,
coalescing, mask and shared layout permit at least a four-byte transaction.
Volatile/scalar/insufficient-width byte requests synchronously fall back with
a warning. INT32 can use a single dword transaction even for strided data.

The current gfx936 DTK compiler cannot lower the unbounded global-to-LDS
intrinsic. Requests that cannot use buffer operations fail with a specific
diagnostic; `is_async=False` remains available. Explicit async loads in loops
also synchronize consumption before reusing their shared buffer.

Synchronous SMEM-to-register-to-HGM transfers are covered. The current DTK
clang-18 gfx936 toolchain treats `llvm.amdgcn.tensor.store.from.lds.d2` and
`llvm.amdgcn.s.wait.tensorcnt` as external calls; a standalone link probe
fails with undefined symbols. This does not establish hardware absence,
but there is no validated executable native reverse async path in this
toolchain.

## Pipelining and LDS limits

On gfx936 the backend uses a **64 KiB LDS budget per CTA**. Peak usage
includes stage buffers, operand layout conversions, and accumulator/output
layout scratch. Tile dimensions, warp count, and stage count all affect
that peak. Large configurations can raise `OutOfResources`; use smaller
tiles or fewer stages and inspect `compiled.metadata.shared` rather than
assuming that a dtype's storage width determines total LDS consumption.
Ordinary FP8 dot can retain FP8 shared buffers and decode in registers before
FP16 MMAC; promotion does not imply that total LDS always doubles.

The FP4 decode prefetch extension applies to `tl.dot_scaled` loops with an
`e2m1` operand, including both scales set to None. It handles single-dot loops
with independent non-volatile global tensor loads and pure conversion/scaling
operations.
It prefetches the data feeding the MMA operands, leaving post-dot E8M0
scales in the compute stage. On the synchronous path, when both MMA layout
conversions have distinct, uniquely identified input loads, it stages those
conversions directly, preserving their shared layouts and packed or decoded
representations. This avoids an additional raw-LDS round trip before the MMA
operand copy; other conversion graphs keep the general copy path. Native
BF16/FP16 MMAC remains enabled.
Loops with stores, atomics, barriers, nested control flow, or loop-carried/load-dependent
addresses retain the existing scheduling behavior. Other graphs are not
covered by the extra FP4 scheduling rule.

Stage buffers add LDS traffic, synchronization, and register live ranges.
Enabling pipelining does not guarantee a speedup for a given shape; benchmark
stage counts, including `num_stages=1`, under the intended workload.

`num_stages > 1` alone does not prove that a loop was pipelined. Precision
regressions check each operand's packed or decoded shape, distinct mutable
stage-buffer allocations/indexing, and native MMAC in the generated IR/ISA.
Pure FP4 regression graphs also reject redundant initialized MMA buffers.
BF16 loaded-accumulator tests cover eight warps and
three/four stages, including the barrier between accumulator scratch reads
and stage-buffer writes. The targeted FP4 and BF16 hardware tests each reuse
the same cached kernel and inputs 500 times, checking all results and guards.

## Validation and measurements

Build the HCU backend and run from the repository root. Select an idle device
for hardware checks:

```sh
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' python -m pytest \
  third_party/hcu/python/test/unit/language/test_precision.py \
  third_party/hcu/python/test/unit/language/test_precision_casts.py \
  third_party/hcu/python/test/unit/runtime/test_precision_cache.py -k 'not values'

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m pytest \
  third_party/hcu/python/test/unit/language/test_precision.py \
  third_party/hcu/python/test/unit/language/test_precision_casts.py \
  third_party/hcu/python/test/unit/runtime/test_precision_cache.py -k values

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python \
  third_party/hcu/python/benchmarks/precision.py --output precision-results
```

The benchmark checks results and guards before timing, records median/p20/p80,
and saves TTIR, TTGIR, LLVM IR and final assembly. It compares tiled FP4/FP8
GEMM with predecoded/prescaled BF16 operands under the same launch options.
Storage widths differ. Vector cases cover 256, 16,390 (masked tail), and
1,048,576 logical elements. Effective bandwidth counts application bytes,
not internal LDS traffic. Static instruction counts include the kernel body;
they are not dynamic instruction counts or peak-throughput claims.

The gfx936 regression covers exact packed-bit checks, conversion boundaries,
nonzero INT32 accumulators and outputs for INT8/INT4 dot, overflow, layouts,
partial tiles, and repeated two-stage loops. Negative tests reject INT32
matrix operands. Software decode and grouped scaling add work compared with
predecoded BF16 operands. Pure-copy async requests with an immediate wait
need not outperform synchronous loads. Measurements do not imply performance
on other shapes or targets.

The precision modules are included by the existing HCU unit-test CI command.
Provider-source cache tests use isolated package copies and explicit gfx936
cross-compilation without starting a device kernel.
