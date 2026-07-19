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
- `triton.language.atomic_mul`: available
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

## FlagTree results

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

## Current acceptance boundary

- Clean XPU build and isolated wheel import pass.
- BMM quick validation passes: 3 passed.
- 18 shared non-passes are baseline limitations, not FlagTree migration
  regressions.
- The two documented `llvm_trust` codegen failures remain open without a
  vectorization workaround.
- `scatter`, `rms_norm`, and `replication_pad3d` remain the three unexplained
  FlagTree-only differences requiring follow-up.
