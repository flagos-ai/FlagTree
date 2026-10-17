# Native KDA arithmetic contract

Current acceptance (2026-10-17): the complete native profile passes official
vLLM greedy8 twice with reset on four H800s (prefix13). All four kernels have
zero spill. The final closure also includes MLA index compaction and Q/KV RMS
association; no CPU gate was relaxed. This acceptance does not claim bitwise
equality for every FP32 intermediate state.

The explicit vLLM profile must retain raw FP32 `A_log` through the constant
island and compute its exponential inside the gated KDA kernel. CPU
precomputation of `exp(A_log)` can differ from the native GPU exponential.
Those rate differences and the K reduction association produce rare BF16
readout differences, first observed at layer 17 on the second position.

The core extension is `decay_input="rate"|"log_rate"` on both
`nn.kda_gated_recurrent_core` and `nn.kda_gated_step`. The existing
`decay_rate` operand remains an FP32 `[1,H,D]` map with the same distribution
and storage ABI. `log_rate` interprets it as raw A_log and evaluates exp at
the kernel's FP32 boundary; `rate` keeps existing semantics and canonical IR.
CPU evaluation applies the same interpretation while retaining its existing
transcendental diagnostic limitations. Effects, tuple results, packing and
head-wise distribution are unchanged. Unsupported values must be rejected.

The selected SM90 `reduction_layout="warp_k"` is a physical schedule. It
groups four consecutive K elements within each thread, with K across the
32 lanes of a warp. L2 sums and state dot products use explicit two-dimensional
numerical layouts. Merely constraining offsets does not constrain reductions.
A one-dimensional forced slice can introduce an expensive reshape; the
matrix reduction keeps the layout conversion legal. The native reduction's
`reshape allow_reorder` retains the thread's original four contiguous values,
so changing the layout to a logically strided vector gives another sum order.

Trials 388/389 calibrate real later-layer inputs and expose the failing
readout. Trials 395 show identical FP32 intermediates under native Triton and
FlagTree when using the same one-warp algorithm. Trial 401 matches all
intermediates in the eight-warp prototype; 402 matches all 32,768 BF16 outputs
and 4,194,304 FP32 state values across four layers and four ranks, with the
native rate. Using the CPU rate still leaves a layer-17 BF16 difference.

Before publication, require checked-in rounding-sensitive fail-before/pass-after
GPU coverage, scalar/packed outputs, default hash compatibility, typed
emission/reload, CPU interpretation and invalid-attribute checks, schema and
distribution neighbors, normal producer packing with compute-body Pack=0,
and full-model zero-spill JIT. The native profile must change producer maps
and consumer interpretation together. Readonly contents change, so kernel-only
source refresh cannot publish this extension: regenerate the artifact through
normal compilation and weight publication. Final acceptance remains complete
45-layer TP4 independent official greedy8 twice with reset.

Core support and the paired native producer/consumer profile are implemented.
The selected independent-reference regression fails before the numerical
layout is supported (403: 13,178 FP32 differences) and passes after (404).
CPU/schema/resume and related importer/device tests pass (408: 126 tests).
Real four-layer/four-rank decode and FlashKDA prefill outputs and states are
exact (410/412). Fresh full compilation passes all Pack gates (407).
The full 45-layer JIT also passes (409: 253 registers, 65,536 shared bytes,
128-byte ABI stack, zero spill). Normal raw-log weight publication also
completed (417); the subsequent prefix10 run failed official greedy acceptance (418).

The complete run (418, prefix10) matches `[3847,284,1373]` and fails at the
fourth token (`10972` versus official `8282`). Four actual rank kernels remain
zero spill. The next isolated readout defect is gated RMSNorm's inverse:
the pinned native Triton kernel computes `1 / sqrt(variance + eps)`, while
the default FlagMega kernel uses approximate `rsqrt`. Trial 435 varies this
choice independently of the profile's sum-layout helper: only sqrt/div
eliminates the layer-9 third-position BF16 difference.

The physical `gated_norm_apply` selection now supports
`inverse_std_algorithm="rsqrt"|"sqrt_div"`; the default stays `rsqrt` and
only the explicit native catalog selects `sqrt_div`. Plain norm operators,
CPU evaluation, tensor types, distribution and readonly contents are unchanged.
The independent checked-in seed-20 regression uses externally supplied GPU
statistics to isolate this inverse boundary: 439 fails before (one BF16 and
523 FP32 elements), 440 passes after. Trial 447 passes 45 related tests;
full JIT passes with zero spill (443: 267.195 seconds). Normal kernel-only
refresh takes 62.850 seconds and preserves all 64 readonly files (444).
The complete prefix11 run matched the first four official tokens and failed
at the fifth (`44979` versus `4218`), leaving official greedy8 twice open at
that stage. Prefix13 closes token acceptance as recorded above. Observed
states still expose a small FP32 layer-36 prefill difference; component
calibrations do not establish full-model state equality.
