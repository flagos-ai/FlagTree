# XPU Triton 3.6 Migration Validation

This document records the clean-build and FlagGems validation performed for
the XPU Triton 3.6 migration. It separates FlagTree-only differences from
failures that also reproduce with the pristine internal Triton baseline.

## Validated revisions

- FlagTree migration head: `8202eb4de3cd1a77acccc945a77e9bed0f2e76dc`
- Clean-build source base: `6691ecb62b1a4caeed56e2d17e755c53b1b2a1ed`
- Pristine internal Triton baseline:
  `d3c64dd65e401239608164d6be4d893b261b4869`
- SDNN prebuilt objects: `v0.3.6.6.0`

## Clean XPU build

The XPU wheel builds successfully from an empty CMake build directory. Select
the built-in XPU backend with `FLAGTREE_BACKEND=xpu`; do not also pass
`third_party/xpu` through `TRITON_PLUGIN_DIRS`.

```bash
FLAGTREE_BACKEND=xpu \
FLAGTREE_CACHE_DIR=/path/to/flagtree-cache \
TRITON_BUILD_DIR=/path/to/empty-build-dir \
MAX_JOBS=32 \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_PROTON=OFF \
TRITON_BUILD_UT=OFF \
TRITON_BUILD_TUTORIALS=OFF \
python setup.py bdist_wheel
```

The host used for validation has system Clang 10 but no compatible `lld`.
Consequently, `TRITON_BUILD_WITH_CLANG_LLD=true` is not valid on that host.
The successful clean build used the default GCC/GNU linker toolchain. The
previous NVWS/TLE generated-API failure did not reproduce when only the XPU
backend was selected; it was caused by including unrelated default backends,
not by an XPU source dependency.

The resulting wheel passed an isolated installation and import check:

- backend marker: `xpu`
- `triton.language.atomic_mul`: symbol available; correct XPU multiply
  atomicity is not established and FP16 is rejected by FlagTree semantic checks
- wheel SHA256:
  `75c5bb4825bbadc4c865d24378862b3fb7b62dcbc289af5325617a080ec45b61`
- installed `libtriton.so` SHA256:
  `75c593b7c9e7b79c0a78dc003aba2db9d49f2c17f365179832e4841b6f598f8d`

## FlagGems test method

The historical regression manifest contains 150 run entries from the earlier
30+60+60 batches. Sixteen entries are repeated, so the manifest represents
134 unique pytest markers. Repeated entries were retained to keep the result
comparable with the historical runs.

Each run used:

- one physical XPU visible to the process;
- logical device 0 selected with `torch.cuda.set_device(0)`;
- `XPU_EVENT_KL3_ENABLE=1`;
- `pytest --quick --ref cpu`;
- an independent HOME, Triton cache, pytest JSON result, and complete log;
- a 300-second per-run timeout.

The run used four idle physical XPU cards. `pytest-repeat` was disabled because
its `repeat` marker conflicts with the FlagGems operator marker. A matching
`tests/test_<marker>.py` file was selected when present; marker selection was
used only when no matching test file existed.

## Initial FlagTree results

| Scope | Pass | Fail | Timeout |
| --- | ---: | ---: | ---: |
| Historical run entries | 127/150 | 21/150 | 2/150 |
| Unique markers | 111/134 | 21/134 | 2/134 |

All 150 entries have a final status and independent log. There are no missing
statuses, collection errors, or runner failures in the final result.

## Internal comparison

All 23 non-passing unique markers were rerun with pristine internal Triton
using the same FlagGems source, test selection, runtime settings, device
isolation, and timeout.

Eighteen markers also failed or timed out with internal Triton and are not
FlagTree-only regressions:

| Category | Markers |
| --- | --- |
| Functional or runtime failures | `div`, `smooth_l1_loss`, `kthvalue`, `lgamma_`, `baddbmm`, `index_reduce`, `mm`, `mode`, `sort`, `normed_cumsum`, `reflection_pad1d_backward`, `upsample_linear1d`, `unique_consecutive`, `grid_sample`, `tril`, `digamma` |
| 300-second timeout | `median`, `pad` |

Five markers passed with pristine internal Triton but failed with FlagTree:

| Marker | FlagTree symptom | Status |
| --- | --- | --- |
| `mul` | Large BF16 broadcast result mismatch | Isolated to `llvm_trust` XPU LLVM codegen |
| `repeat_interleave` | Large BF16 result mismatch | Isolated to `llvm_trust` XPU LLVM codegen |
| `scatter` | 30 functional failures and 9 passes | FlagTree-only follow-up |
| `rms_norm` | Backward input gradient near zero; 63/64 values mismatch | FlagTree-only follow-up |
| `replication_pad3d` | Two XPU pass/resource compile failures and two passes | FlagTree-only follow-up |

The `mul` and `repeat_interleave` failures are documented rather than bypassed.
TritonXPU vectorization remains enabled. See
[`llvm-codegen-validation.md`](llvm-codegen-validation.md) for the same-IR,
dual-toolchain, crossed `llc`/`elfconv`, and injected-XPUBIN evidence.

## Final clean A/B update

A later clean run executed all 150 historical entries in both isolated
environments instead of rerunning only the initial FlagTree failures:

| Environment | Pass | Fail | Timeout |
| --- | ---: | ---: | ---: |
| FlagTree | 127/150 | 20/150 | 3/150 |
| Pristine internal Triton | 130/150 | 18/150 | 2/150 |

There were 126 entries passing in both environments and 19 shared non-passing
entries. `rms_norm` and `replication_pad3d` passed the final manifest after the
compatibility fixes and are no longer unexplained FlagTree-only differences.
The remaining confirmed FlagTree-only differences are FP16 scatter multiply,
whose frontend difference must not be separated from the shared non-atomic XPU
multiply lowering described below, plus the BF16 broadcast `mul` and BF16
`repeat_interleave` failures isolated to `llvm_trust` code generation.

`cross_entropy_loss` failed in the first FlagTree run and passed with one fresh
cache, but later controlled testing ruled out a simple cache-hit artifact. The
failing unweighted backward case failed 6/20 times with one shared warm cache
and 3/10 times with independent cold caches. A single byte-identical compiled
binary alternates between pass and fail, and failures leave contiguous
zero-valued holes in the second gradient row. Pristine internal Triton then
reproduced the same failure during an isolated IR-dump run. FlagTree and
internal TTIR have the same computational graph; TTXIR differs only in debug
paths and lowers the output identically through `GM2LM -> select -> LM2GM`.
Cross entropy is therefore classified as a shared XPU nondeterministic
correctness issue. The masked-memory simulation settings used for `grid_sample`
did not resolve it (5/20 failures).

`instance_norm` failed in the first internal run but passed with a fresh cache.
Follow-up fixed-input testing passed 50/50 times in each tree for shape
`(2,1,2,1)`. A 200-seed deterministic comparison produced the same 17 small
input-gradient tolerance mismatches in each tree. It is classified as shared
small-shape numerical sensitivity under the current strict tolerance, not a
FlagTree-only or cache-only regression.

`grid_sample` was a functional failure in both environments, but the complete
quick test timed out only with FlagTree. This timeout/runtime difference remains
open even though the masked-memory correctness root cause is shared.

## Shared XPU atomic-multiply limitation

The presence of `triton.language.atomic_mul` does not imply correct atomic
multiply semantics on XPU:

- FlagTree rejects FP16 `tl.atomic_mul` during semantic type checking.
- Internal Triton accepts FP16, but both trees lower FP16/FP32 multiply through
  synchronous `GM2LM -> mul -> LM2GM`, without a hardware atomic instruction or
  CAS retry loop.
- Duplicate-address contention tests produce nondeterministic incorrect results
  in internal FP16/FP32 and FlagTree FP32.

Do not resolve the FlagTree frontend difference by only allowing FP16. A valid
fix requires a native atomic lowering or a correct CAS loop and must be tested
with duplicate-address contention.

## AABS square-tile fallback limitation

The XPU AABS-adjusted `8x8` and `16x16` square configurations can fail in
`TritonXPULegalize` with mixed tensor-type verifier errors. The runtime
autotuner now treats an all-`inf` adjusted timing result as an AABS failure,
retries the original metadata, and restores the successful original
configuration. This makes affected operators such as `replication_pad3d`
usable, but the underlying square-tile verifier defect remains open. The
fallback is not evidence that the adjusted configuration compiles correctly.

## Current acceptance boundary

- Clean XPU build and isolated wheel import pass.
- BMM quick validation passes: 3 passed.
- The final clean matrix has 19 shared non-passing entries; they are baseline
  limitations, not FlagTree migration regressions.
- The two documented `llvm_trust` codegen failures remain open without a
  vectorization workaround.
- The shared atomic-multiply semantics, AABS square-tile verifier defect,
  `grid_sample` masked-memory lowering, 3D `grid_sample` vectorize failures, and
  FlagTree-only `grid_sample` timeout difference remain open.
- `cross_entropy_loss` is a shared XPU nondeterministic correctness issue, not a
  cache-hit artifact. Its FlagTree/internal TTIR and TTXIR output lowering is
  semantically identical.
- `instance_norm` is classified as shared small-shape numerical sensitivity;
  fixed-input and deterministic-seed tests match between both trees.

## Shared XPU `grid_sample` limitation

Follow-up A/B investigation confirmed that the 2D `grid_sample` failure is
shared by FlagTree and pristine internal Triton. It is not a FlagTree migration
regression.

The failing path uses the medium/large-output tiled kernels. Small 2D kernels
pass, while tiled nearest and bilinear kernels intermittently leave valid
output elements as zero. The mismatch count and spatial pattern change between
runs in both compiler trees. Enabling automatic core tiling does not resolve
the problem.

The failure is caused by the legacy XPU masked-memory lowering:

- A masked load does not reliably preserve the Triton `other=0` semantics.
- A masked store is lowered through coarse-grained DMA and cannot precisely
  guard all lanes of a 2D tile. Tail lanes can affect physically adjacent
  output locations.

Two equivalent compatibility configurations eliminate the 2D failures in
both trees:

```bash
# Legacy-compatible simulation paths
export TRITONXPU_OTHER_SIM=1
export TRITONXPU_STORE_MASK_SIM=1

# Or, with a sufficiently new XRE and the required DMA mask setup
export TRITONXPU_IS_USE_MASK_ZERO=1
```

`TRITONXPU_IS_USE_MASK_ZERO=1` must not be enabled globally without confirming
the runtime prerequisites. The backend warns that it requires XRE newer than
`5.0.21.37` and the corresponding `dma_excp_mask` configuration.

Validation evidence:

- FlagTree and internal each passed 100 repeated nearest and bilinear tiled
  invocations, 205 cases per tree, with `TRITONXPU_IS_USE_MASK_ZERO=1` and zero
  mismatches.
- FlagTree's 4D `grid_sample` quick suite passed 82 tests with that setting.
- `TRITONXPU_OTHER_SIM=1` plus `TRITONXPU_STORE_MASK_SIM=1` also produced zero
  mismatches on both trees in repeated runs.

The separate 3D result is now fully classified. Zeros-padding nearest and
trilinear small cases pass in both trees. Border and reflection kernels fail in
the same `TritonXPUVectorize` pass in FlagTree and pristine internal for all
five configs, including `BLOCK_SIZE=256`. Diagnostic closure of vectorization
allows compilation and execution; the reference comparison then reaches the
independent XPU `XDNN_UNIMPLEMENTED grid_sampler_3d` limitation. This is a
shared pre-LLVM compiler limitation, not an `llvm_trust` issue or a FlagTree
migration regression. Deeper vectorizer bisection is deferred to a separate
shared compiler task.

The FlagTree-only full quick-suite timeout is also explained rather than
unclassified. `256x256 border bicubic` spends 64.38 seconds in AABS dependency
analysis without adjusting any config. Its total autotune time is 78.49 seconds
versus 14.07 seconds in internal; `FLAGTREE_AABS=0` reduces FlagTree to 13.52
seconds with the same selected `BLOCK_SIZE=512`. AABS optimization is deferred,
and the disable flag remains diagnostic-only.

## Reduction scratch allocation alignment

IR comparison found a real but non-blocking codegen difference in reduction
scratch layout:

- FlagTree uses cumulative non-overlapping regions such as `0/1040/2080`.
- Internal reuses the `1040` region for later sequential reductions.

This difference is not the cause of the cross-entropy gradient-hole failure:
FlagTree fails with the conservative non-overlapping layout, and internal fails
with the reused layout. The policies have different tradeoffs rather than both
being known-bad. FlagTree avoids reuse hazards but consumes more scratch memory;
internal is more resource-efficient. FlagTree has now been aligned with the
internal reuse behavior in `ReduceOpHelper` and `ScanLoweringHelper`. Fresh
codegen validation shows the expected `0/1040` reuse instead of cumulative
`0/1040/2080` bases. The cross-entropy quick repeated run was `21 pass / 9
fail`, consistent with the earlier intermittent baseline, and RMS norm remained
`0 mismatch` for output, `dx`, and `dw`. The quick reduction/scan suite (`sum`,
`mean`, `amax`, `argmax`, and `cumsum`) also passed all 53 cases. The alignment
is therefore validated as a layout change, but is not a correctness fix for the
shared zero-hole failure.
