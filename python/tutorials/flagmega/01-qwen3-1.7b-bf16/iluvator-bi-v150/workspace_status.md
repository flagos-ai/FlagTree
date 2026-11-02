# BI-V150 workspace checkpoint, 2026-10-28

This checkpoint records the Triton/TLE optimization work and outstanding
compiler investigations. It is not a claim of performance parity or a fully
passing Iluvatar test suite.

## Launch and numerical contract

C++ launches 16 CTAs with 1024 physical threads each, independently of the
shared descriptor's former 32-warp value. The tutorial's Triton target now
uses 16 physical 64-lane warps per CTA and the same 16-CTA grid. Explicit
register/lane layouts retain the original Q/K/V, GEMV, GLU and RMS reduction
trees after this launch change. Strict floating-point contracts disable
fusion for generated runtime launches and native code generation.

The complete 28-layer model, 151936 vocabulary entries and canonical
4063488256-byte BF16 weight payload remain the measured workload.

## Code and runtime changes

- BI-V150 templates use warp-owned flat SME transfers with explicit FIFO
  waits, shared-slot draining and verified packed-weight/source addresses.
  Dense projection and logits kernels distribute independent N/K fragments;
  normalization and attention combine batch independent outputs.
- The readonly-layout selector proves complete canonical byte ranges and
  every reader before changing device addresses. The runtime permutes a
  freshly loaded pool once before preparation or graph capture. C++ retains
  its existing schema over the shared byte validator.
- TLE lowering preserves explicit layouts through conversion, coalescing,
  canonicalization and rematerialization. Shared conversion scratch cannot
  reuse an arena ahead of pending flat-SME completion.
- Convergence attributes propagate through collective device wrappers and
  callers. This removes the demonstrated lane-predicate duplication of SME
  calls. Pipe lowering also retains live captures and derives drain
  participants from writer/reader tasks.
- COREX advertises its lack of the CUDA cooperative-launch API. Mesh grid
  metadata honors that capability, and a unit launch axis has coordinate zero.
- Regression tests cover layouts, strict FP32 products, SME bytes/ownership,
  masked tails, readonly consumers and constant/dynamic function-pool rows.

## Performance and artifact identity

The retained v47 measurement reports C++ 6.101/8.202 ms and Triton/TLE
6.578/8.770 ms at contexts 1/2048. Across six contexts, this is 38.5–41.5%
below the aligned Triton baseline, with a remaining 6.9–8.2% latency gap
relative to C++. Device-graph timing includes the complete model and identical
argmax; it differs from the vLLM request and chat CLI timing boundaries.

The measured v47 package, renderer v21, has generated source SHA-256
`239f379e2f11eb03cd95470fc797a4054f587fdace743a2520cad301500d88cc`.
The restored package passed exact comparison of every logit at all 2112
sequential steps. The current v22 renderer reproduces the unpromoted v48
gate/up layout experiment, not the measured v47 package. Current-renderer
output must not inherit the v47 figure's performance claim.

The SVG is retained. `**/figures/data/` is now ignored, and the 184 previously
tracked data files are removed from the Git index while all local bytes remain
intact. Raw reports, diagnostic bundles and archived benchmark sources in
that directory are local records and are not distributed by a fresh clone.
Links into that directory in the tutorial require those local records.

## Unresolved private-frame defect

The checked-in [minimal case](../../../../../third_party/iluvatar/python/test/unit/tle/test_call_frame.py)
contains no model, weights, SME operation, grid barrier or CUDA graph.
One CTA with 64 threads is sufficient: direct and nested pressure cases each
overwrite 128 of 4096 values; controls preserve every value and all guards.
The original 16-CTA x 1024-thread cases each overwrite 32768 of 1048576 values.

An independent main checkout at `2f1229e51c03b345a1c72207854828eed7282de9`,
built with main's pinned LLVM22 v0.6.1, reproduces four failures and four passes
without modifying tracked compiler sources. The separate single-warp command
reports two failures and two passes. See the
[reproduction note](../../../../../third_party/iluvatar/python/test/unit/tle/call_frame_repro.md).

MIR and register traces identify per-thread/per-warp byte-unit disagreement
in entry stack setup, caller restoration and non-entry frame adjustments.
A 272-byte logical per-thread frame needs 17408 bytes of physical warp
storage; the generated stack pointer uses 272. The relevant target sources
are `BIFrameLowering.cpp`, `BIMachineFunctionInfo.cpp` and
`BIRegisterInfo.cpp`. A source-level fix remains outstanding. Binary edits
used to establish causality are diagnostic evidence only and are not shipped.

## Checkpoint validation

Fresh checks on GPU 1 before upstream integration:

- Target, runtime, readonly ABI, GEMV, attention and distributed checks:
  544 passed, 20 skipped and 24 failed. Twenty failures were NVIDIA tensor-map
  or SM90 requirements; four were attention head-resharding results.
- Replacing only the combine template with the original committed template
  reproduces all four attention failures. The newly fetched upstream includes
  both the head-address repair and NVIDIA test-device guards.
- Native explicit-layout, FP32 and flat-SME/convergence checks: 38 passed.
- The wider pipe/copy/warp-specialization attempt reached additional failures
  and stalled before completing. It was terminated and supplies no passing
  suite result. A concurrent private-frame rerun did not complete either;
  the earlier independently verified main result remains the valid evidence.

Raw logs for these checks are under `.local/publish-20261028/`. Post-integration
results are recorded below before pushing. Unfinished diagnostics and
architecture-specific unsupported tests must not be represented as a green
full-suite result.

## Upstream integration and final validation

The checkpoint `2a64aae18` is integrated with the seven upstream commits
through `254312d22`. Conflict resolution preserves the BI-V150 reduction
layouts and readonly device packing alongside AMD weight contiguity hints,
prefetch planning and scoped compilation knobs. The strict floating-point
contract remains active inside the new compilation scope. Four local test
fixtures now provide the upstream prefill row count and attention/statistics
owner-coordinate fields; the tested shapes and numerical expectations are
unchanged.

`./build-iluvator.sh` rebuilt and installed the editable Iluvatar extension
successfully. Final validation with that extension reports:

- Focused runtime, target, readonly ABI, GEMV, GLU, gather, attention,
  function-pool, AMD planning and distributed checks on GPU 4:
  **636 passed, 43 skipped, 7 failed**. One failure requires tensor maps that
  the active driver does not support. Six source-backed CUDA cases fail
  readonly host-page registration with driver status 1 in
  `runtime/source_rdata.py::_RegisteredFile`; their CPU cases pass. The four
  attention head-resharding cases pass after the upstream address repair.
- The four updated fixture suites: **109 passed**, including numerical
  comparisons between the original and aligned reduction layouts.
- Native layout, FP32 and flat-SME/convergence checks on GPU 6:
  **38 passed**.
- The private-frame reproducer on GPU 5: **4 failed, 4 passed**, with the same
  128-value and 32768-value corruption and intact guards as independent main.
- The barrier codegen check with `-x`: **1 failed** at
  `test_tle_barrier_codegen.py:46`. Generated LLIR lacks the expected warp
  lane-mask expression; the generated completion loop also needs further
  investigation. This attempt does not establish a passing pipe suite.
- AMD split-K templates render and parse with K contiguity hints retained,
  with prefetch enabled and disabled. AMD GPU execution was not tested here.
- Conflict-marker, Python syntax and Git whitespace checks pass. No file
  under `**/figures/data/` remains tracked; local raw files are preserved.

Final logs and the fresh private-frame IR/binaries/results remain in the
ignored `.local/publish-20261028/` directory. This push is a workspace
checkpoint with explicit unresolved defects, not a completed compiler fix
or a promotion of the newer renderer's performance.
