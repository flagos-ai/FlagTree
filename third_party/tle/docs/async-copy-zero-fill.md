# Zero-fill proofs for tiled asynchronous copies

[Issue #1586](https://github.com/flagos-ai/FlagTree/issues/1586) is a compile
failure caused by inconsistent transport selection for two copies on one pipe.
The first copy has a mask and the default zero fill; the second is unmasked.
For a logical shared-memory extent of `[80, 256]`, the planner splits both
copies into `[16, 64]` tiles. The masked load's fill becomes
`convert_layout(extract_tile(zero))`.

The regression-introducing decision was the fill guard added by #1429 in
`TleOptimizeLocalPointerAsyncStores.cpp::matchAsyncStoreCandidate`. Its original
`isZeroConst` predicate only recognized a direct constant. The subsequent
`CopyFill.h::hasZeroFillBits` helper recognized layout conversions but still
stopped at an extraction. The masked copy retained ordinary load/store
transport, while the unmasked copy became `cp.async`.
`TleLowerPipeToNvws::recordDataTransport` then correctly rejected the second
commit because the pipe's full-barrier count is a per-pipe contract.

`hasZeroFillBits` now follows both `ConvertLayoutOp` and `ExtractTileOp`, in
either order and through repeated wrappers. Both preserve the element bits of
a splat. The final proof still requires a constant splat with all-zero element
bits. Floating-point negative zero, nonzero constants, and unproven dynamic
values remain ineligible for this load/store-to-async rewrite.

The change is limited to the TLE-owned proof helper and its regression tests.
The shared `isZeroConst` utility, logical tiling, pipe barrier contracts, and
LLVM fill handling are unchanged. The intended optimization remains direct
global-to-shared asynchronous copies; disabling that optimization or removing
the fill guard would not correct the causal defect.

## Regression checks

The minimal pass-level test is
`third_party/tle/test/GPU/local-pointer-async-copy-tiled-fill.mlir`.
It checks extracted floating zero, layout-wrapped extracted zero, nested
integer-zero wrappers, and rejection of negative zero, nonzero, and dynamic
fills. `test_tle_lower_pipe_to_nvws_errors.mlir` also checks that genuinely
different local-store and async commit transports remain rejected.

The Hopper runtime reproducer is self-contained in
`python/test/tle/unit/test_async_copy_logical_zero_fill.py`:

```sh
pytest -q python/test/tle/unit/test_async_copy_logical_zero_fill.py
```

It uses a fresh cache for each case and tests both `N=80` and the `N=128`
control. Source addresses are in bounds for both. A WGMMA consumer sums the two
copies; rows below 73 contribute twice and the remaining logical rows once.
The BF16 inputs are compared with an FP32 reference using `atol=rtol=1e-3`.
Before the fix, `N=80` fails with the mixed-transport diagnostic; after it,
both cases must compile and pass numerically. The separate MLIR test checks
that the matcher retains the async-copy optimization, independently of later
layout legalization or transport lowering.

Neighboring coverage is provided by `local-pointer-async-copy-fill.mlir`,
`test_async_copy_fill_bits.py`, and `test_tle_gpu_async_copy.py`. The helper has
one caller, the local-pointer async-store matcher. Other backend uses of the
native zero predicate are not broadened by this change.

## Independent PR checkpoint (2026-11-10)

The extracted #1538 head `0399b221b1bc72f2cf5d116dce28e1a7772d1ffa` was built
from source with CMake, enabling NVIDIA and AMD, using Python 3.10 and LLVM
22. The helper was then changed and the compiler rebuilt. Runtime checks used
an NVIDIA H800 with PyTorch 2.10.0+cu128.

- Before the fix, the new runtime test reported one failure (`N=80`, the
  original mixed-transport diagnostic) and one pass (`N=128`). The new MLIR
  test failed all three positive-zero conversion checks.
- After the fix, both runtime cases and all six new MLIR cases passed. The
  48 existing exact-fill runtime cases also passed. The explicit pipe
  mixed-transport diagnostic test passed before and after.
- The complete TLE lit target changed from 119 passed / 9 failed to
  120 passed / 8 failed. Only the new tiled-fill test changed status.
  The remaining pre-existing failures were `test_tle_accelerate_matmul_chain_dot`,
  `test_tle_inline`, `test_tle_lower_wgmma`,
  `test_tle_optimize_local_pointer_async_stores`,
  `test_tle_reduce_data_duplication_remote_if`,
  `test_tle_remote_atomic_splat_ptr_lowering`,
  `test_tle_remote_call_arg_store_lowering`, and
  `test_tle_wgmma_pipeline_accumulator_chain` (all under `GPU/`, `.mlir`).
- The broader GPU selection reported 64 passed / 2 failed. Both failures,
  `test_two_group_async_copy_preserves_explicit_waits_and_executes` and
  `test_qkv_bf16_1x256_copy_legalizes_and_executes`, also failed on the saved
  baseline compiler with `copy operation failed: name 'ir' is not defined`.

These independent results verify the #1586 fix and distinguish it from the
remaining extracted-stack failures. They do not replace full-stack acceptance
or establish a performance improvement.
