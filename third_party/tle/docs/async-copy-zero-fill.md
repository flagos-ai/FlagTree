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

## Independent rebased checkpoint (2026-11-16)

The PR is rebuilt independently on main `c5695316` with NVIDIA and AMD enabled.
All five added MLIR regression files pass. The full TLE suite passes 127/133;
the six remaining failures reproduce on main. Native analysis/GPU checks pass
99/108 with the same nine failures as main. The combined H800 and host
selection passes all 216 cases, including all 48 exact-fill cases, both tiled
zero-fill cases, explicit transport ownership, and 13 logical-TMA cases.

Two additional defects in the extracted PR were reproduced and repaired:

- `gpu/core.py::copy` used `ir.value()` for absent mask/fill handles without
  importing `ir`. Both unmasked explicit-copy runtime cases failed before
  the import and pass afterward.
- The default TMA frontend inserted an eviction argument into existing
  positional calls, and `triton_tle.cc::create_tma_copy` moved the logical
  shape from the sixth position. Strict legacy-builder cases fail before
  the fix (8 failures, 4 passes), and a real six-argument binding reproducer
  fails with incompatible arguments. Default calls now keep their original
  arity; non-default eviction is a named argument. The logical-shape overload
  retains its sixth argument and accepts eviction afterward. All regressions
  pass, while explicit policies and required-copy checks remain enforced.

The binding reproducer is
`test_tle.py::TestTmaCopyBarrierFrontend::test_tma_builder_preserves_positional_logical_shape`.
The strict default-call cases are in the same class. The pipe async-store
fixture is aligned with main's zero-result `tle.pipe.create` contract; its
transport assertions are unchanged. Non-NVIDIA tests skip NVIDIA-only PTX
checks using backend and TLE capability checks. All four native source projections
remain identical to main when `__TLE__` is disabled. No runtime speedup is
claimed by this correctness validation.

CommonIR negative-contract checks also reproduce two failures before the final
fix: explicit async transport and eviction policy reached ordinary copy
emission. CommonIR now rejects both unsupported contracts before emitting IR.
The two added cases, the complete 123-case host/frontend module, and both
unmasked explicit-copy runtime controls pass in the final 125-case rerun.
Together with the preceding 216-case selection, 218 distinct cases are covered.


## Backend schema regression (2026-11-16)

The first required HCU CI builds of the rebased PR failed in
`TleLowerTmaCopy.cpp::TMACopyLowering::matchAndRewrite`: the newly added
`TMACopyOp::getEvict()` calls assumed the NVIDIA operation schema. HCU defines
its own `TMACopyOp` without that field. This is a build-time defect, before
any model or kernel tests run.

A minimal reproducer is the TMA lowering object in an HCU-configured build:

```sh
cmake -S . -B build-hcu -G Ninja -DFLAGTREE_BACKEND=hcu \
  -DTRITON_CODEGEN_BACKENDS=hcu -DTRITON_BUILD_PYTHON_MODULE=OFF \
  -DTRITON_BUILD_PROTON=OFF -DTRITON_BUILD_FLAGPRISM=OFF \
  -DLLVM_DIR="$LLVM_SYSPATH/lib/cmake/llvm"
cmake --build build-hcu --target \
  third_party/tle/dialect/lib/Transforms/CMakeFiles/TritonTLETransforms.dir/TleLowerTmaCopy.cpp.o
```

The local C++17 build reproduces both missing-getter errors before the fix
and compiles the same object after it. Lowering now reads the actual MLIR
attribute through `Operation::getAttr`, which covers inherent properties and
attribute dictionaries. A schema without the field retains the default normal
policy. A non-default policy is rejected if the registered backend operation
does not declare the field; it is never silently dropped.

`test_tle_tma_copy_eviction.mlir` covers default policy, both explicit policies,
implicit and user-owned barriers, and rejection of policies on TMA stores.
The six-file final MLIR selection passes. The NVIDIA/AMD build and the final
TMA/frontend/runtime selection also pass. HCU hardware execution is validated
by required CI, not by the local NVIDIA machine.


The MThreads required unit suite then reported one stale public-signature
assertion (12,287 cases passed). The MThreads and Enflame copies of the TLE
API tests both retained the old `gpu.copy` parameter list. Both existing
host tests fail before updating their declared contract and pass afterward.
They now check the added parameters and their backward-compatible defaults:
`is_async=False` and `eviction_policy=""`.
No backend implementation or runtime behavior is changed by this test update.


## Final main sync (2026-11-16)

The final change is rebased onto main `f034f059`, incorporating its TLE API
cleanup and AMD BlockPingpong fix. The NVIDIA/AMD incremental build passes
in 38 seconds. The final selection passes 134 frontend/H800 cases, 26
capability cases, and both backend signature regressions (162 distinct cases).
All six added MLIR files and the TMA/cache-policy/barrier checks pass; the
upstream AMD BlockPingpong regression also passes. All four native non-TLE
source projections remain byte-identical to this main.

The TLE/NVIDIA/HCU C++ sources are unchanged from the previously validated
schema-fix checkpoint; the only C++ change brought by this main sync is the
upstream AMD fix. The earlier HCU compile fail-before/pass-after evidence and
correctness checkpoints above remain applicable. The full-suite baseline
counts above were measured on `c5695316`; the final main sync is verified with
the targeted selections and required CI. No performance improvement is claimed.

## Source/shared layout and TMA policy regressions (2026-11-17)

Two regressions reproduce on the split NVIDIA head `148bc2eb` with an
assertions-enabled compiler and an SM90 GPU. Run the checked-in reproducers:

```sh
pytest -q python/test/tle/unit/test_async_copy_layout_compatibility.py \
  python/test/tle/integration/test_tle_tma_eviction.py
```

`TleDowngradeInvalidAsyncCopy.cpp::hasLegalCpAsyncWidth` and
`getRequiredAsyncCopyLayout` considered global contiguity without checking the
register-to-shared mapping. A column-major `4 x 32` BF16 source copied into a
row-major shared destination was accepted as a four-byte transaction, even
though its adjacent source elements are separated in shared memory. NVIDIA
LLVM lowering reduced the vector to two bytes and hit its width assertion.
Both existing-layout validation and candidate-layout legalization now cap the
vector width using the composed register-to-shared linear layout. The shared
layout comes from the actual memory descriptor, including its allocation
shape. Unrepresentable explicit copies receive the required-copy diagnostic;
representable row-major BF16, matching column-major BF16, and scalar FP32 copies
still execute asynchronously with four and eight warps. The diagnostic test
fails before this fix.

`TlePlanLogicalDomains.cpp::applyRootRewrite` rebuilt logical TMA copies
without their eviction attribute. Dense `64 x 256` FP16 loads reached
`AsyncTMACopyGlobalToLocalOpConversion::matchAndRewrite`, which rejected
non-default policies, while logical `80 x 256` and `96 x 256` loads silently
lost them. Each physical copy now inherits the actual MLIR eviction attribute.
The TLE-gated NVIDIA conversion uses the existing L2 policy builder and emits
the `cp.async.bulk.tensor` cache-hint qualifier and its 64-bit policy operand.
Default-policy emission and the non-TLE source are unchanged. Generic attribute
access preserves compatibility with HCU's distinct operation schema.

Before the fix, the TMA test has six failures and three passing default-policy
controls: two dense compile failures and four tiled PTX-policy failures. After
the fix, all nine variants execute with exact WGMMA results; every non-default
TMA instruction carries the requested PTX policy. The test inspects PTX text
and numerical outputs. It does not claim a runtime speedup from an eviction hint.
