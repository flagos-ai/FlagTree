# FlagMega workspace checkpoint, 2026-10-21

Branch: `feature/flagmega`. The workspace was based on `8da3816de`; its changes
were committed as `929957c77`. Merge `85eb50cb6` integrates remote head
`f3c531a47`, preserving the 27 earlier unpublished local commits and the three
new remote commits. The remote additions cover AMD RDNA TLE lowering, AMD
gfx1100/gfx1201 targets and the NVIDIA sm89 target. The merge had no conflicts.

## Saved work

The earlier local commits add and tune the BI-V150 C++ SME implementation,
packed BF16 accumulation and Triton GEMV tile schedules. This checkpoint adds:

- Flat SME FIFO counter waits and consumer shared-read draining; fused QKV
  staging; ordered GLU batch issue; overlap of the large-N scalar tail with
  complete idle warps; and COREX load-latency scheduling parameters.
- Proven GLU/down readonly device permutations, preserving canonical IR and
  artifact bytes. All consumers must share the selected address contract, and
  the complete layout record set is validated before any device bytes change.
- Shared arena sizing, disjoint statistics/argmax regions and QKV reader
  lifetime synchronization. Exit barriers merge with schedule barriers only
  when every enclosing caller establishes the required ordering.
- Four-head paged-attention staging with 64-token KV loads, query hoisting,
  canonical 16-token softmax updates and explicit FMA. Shared storage is
  116992 bytes within the existing 128 KiB budget.
- A vLLM adapter that combines single-request decode input preparation into
  one pinned-buffer upload and one GPU kernel. It retains native prefill,
  scheduling, sampling, noncontiguous page tables and asynchronous token
  ownership, with no KV copies or native-model decode calls.
- Native vLLM FULL CUDA Graph, FlagMega-vLLM and standalone chat CLI comparisons.
  All three variants run in serially rotated rounds, with a fresh process per
  variant/round and the same scenario start-temperature limit of 70 degrees C.
- Three H800-style figures: decode latency, decode throughput and complete
  request throughput. The BI-V150 tutorial README is entirely English.
  Complete raw reports, rejected candidates, source/library identities,
  before/after regressions and temperature/clock diagnostics are retained.

The implementation changes are under FlagMega's C++ codegen/runtime, its tests
and the BI-V150 tutorial. The remote AMD/native compiler changes are preserved
by the merge. The `third_party/triton_shared` gitlink is unchanged. Ignored
`.local/` experiments, model weights, generated shared libraries,
`build-iluvator.sh` and `dist/iluvatar/` are excluded from the checkpoint.
Large artifact verification JSON reports are stored as gzip with hashes of
their original uncompressed contents.

## Performance and timing boundaries

The final comparison uses one BI-V150, the complete 28-layer Qwen3-1.7B model,
BF16, batch/concurrency one, no prefix caching, maximum context 3072,
16 KV pages of 256 tokens and 64 generated tokens. Each variant has three
rounds, one warmup and five measured requests per prompt length per round;
all 945 measured decode intervals per length/variant are retained.

| Prompt tokens | Native vLLM FULL graph | FlagMega-vLLM C++ | Standalone chat CLI | Native / C++ speedup |
|---:|---:|---:|---:|---:|
| 32 | 11.812 ms | 7.231 ms | 6.585 ms | 1.63x |
| 128 | 11.901 ms | 7.300 ms | 6.688 ms | 1.63x |
| 1024 | 12.798 ms | 8.301 ms | 7.704 ms | 1.54x |
| 2048 | 13.630 ms | 9.444 ms | 8.825 ms | 1.44x |

These are host-visible decode medians, excluding the first output token.
Throughput is computed from each complete interval sum or request duration;
it is distinct from the inverse of median latency. CLI's token-scan prefill
has higher TTFT, so its complete request throughput falls to 8.36/3.99 tokens/s
at prompt lengths 1024/2048 despite faster decode. CLI output is checked against
an independent native token-scan reference; its BF16 sequences differ from
native batched-prefill output. The figures explicitly state that boundary.

A separate 32-token diagnostic uses means: native `engine.step` is 10.606 ms,
graph device time 7.355 ms and the outside-graph residual 3.252 ms. Native's LM
head contributes 0.957 ms outside its graph; the remaining outside-model
residual is 2.294 ms. C++ includes the LM head inside its graph: `engine.step`
7.228 ms, graph 6.107 ms and residual 1.121 ms. These residuals include CPU/GPU
overlap and cannot be interpreted as isolated CPU time or subtracted from the
chart medians.

Earlier five-family isolated diagnostics reached 80.0115%-80.0878% net
bandwidth utilization with a fixed 819 GB/s denominator and five shell-time
subtractions. That metric covers GLU, down, QKV, attention output and logits;
startup/device reordering is excluded. It does not measure complete decode
utilization. Source identities and raw measurements remain attached to their
respective experimental stages.

See the [English tutorial](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/readme.md)
and [figure data documentation](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/README.md)
for reproduction commands, exact identities and full results.

## Root causes and regression evidence

The newly expanded large-N SME selection omitted the instruction's K-group
address alignment constraint. Unpadded N=549/1023 produced incorrect front
elements while the scalar tail remained correct. Selection now proves
64-byte group-offset alignment; aligned workloads retain SME. Two failing
GPU cases and 12 passing neighboring-layout checks are recorded.

Fused QKV output reducers could still read the shared arena when a following
consumer overwrote it. A delayed-reducer GPU test fails with the exit CTA
barrier removed and passes with the required synchronization. The descriptor
only merges that barrier after proving all enclosing call paths. Layout,
FIFO, QKV-only arena and disjoint argmax checks cover neighboring uses.
See [SME validation](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/sme-validation/README.md).

The original serial-head attention limits long-context speedup. The staged
implementation retains canonical arithmetic: a rejected wider softmax update
changed numerical state, and query-hoisting exposed implicit FMA contraction
differences. Explicit FMA and 16-token state updates preserve the numerical
boundaries while keeping cooperative 64-token loads. GPU tests cover paged
tails and unaligned cache bases; final full-model validation checks every
logit over 2112 steps. See [attention evidence](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/attention-root-cause.zh.md).

The vLLM plugin's `ModelRunnerFL._prepare_inputs` submitted multiple uploads
and index/cast/copy operations per decode step. Fused input preparation reduces
this overhead while preserving the existing buffer/event ABI. The recorded
performance gate fails before the change and passes afterward. See
[input-preparation evidence](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/root-cause.zh.md).

The earlier CLI comparison mixed a continuously running, globally prewarmed
CLI process with fresh vLLM processes. Long token-scan prefill heated the GPU;
in the same process graph execution rose from 6.134 to 8.430 ms and returned
to 6.528 ms after cooling, while graph-external overhead stayed near 0.100 ms.
The comparison now validates process lifecycle and identical scenario start
conditions. The rejected shared-process comparison has fail-before/pass-after
tests. Continuous hot-state behavior remains documented; no hardware/driver
fix is claimed. See [measurement audit](../../python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/chat-cli/measurement-audit.zh.md).

## Checkpoint verification

On idle GPU 8 in the COREX-matched `flagtree` Conda environment:

- Before merging, the complete C++ codegen suite and four changed tutorial test
  files passed **136 tests in 29.50 s**.
- After merging, the same tests plus serving, BI-V150/AMD target routing,
  artifact-device capability, sm89 resource contracts and portable split-K
  template resolution reported **213 passed, 2 failed in 31.18 s**.
- Both failures reproduce on a detached checkout of pre-merge `929957c77`:
  **2 failed in 3.36 s**. The relevant test, distributed-launch helper and
  tutorial target have identical Git blobs before and after the merge.
- All three figures reproduce byte-for-byte from saved raw reports. Six formal
  raw-report hashes, two artifact-verification hashes and twelve SME evidence
  hashes were checked. `git diff --check` passes.

[Verification logs](workspace-20261021/manifest.json) contain uncompressed
SHA-256 identities for the three compressed checkpoint test logs.

The two existing target failures remain unchanged:

1. `test_corex_mesh_barrier_does_not_request_unsupported_cooperative_launch`:
   `python/triton/experimental/tle/language/distributed.py::_apply_mesh_grid_launch`
   sets `launch_cooperative_grid=True` for the test's COREX options, while the
   test expects `False`. Runtime/driver behavior needs separate investigation.
2. `test_bi_v150_tutorial_target_selects_async_qkv_pipeline`: the test expects
   experimental async QKV by default, while `local_optimizations/target.py`
   selects it only with `FLAGMEGA_QKV_ASYNC=1`. This checkpoint retains the
   documented current selection and does not change assertions or enable the
   unvalidated pipeline.

Prior model validation recorded complete native/C++ token agreement for
144 requests (9216 output tokens), 14 adjacent sampling requests per variant
and bit equality for all 151936 logits over 2112 steps. CLI's 72 warmup/measured
requests match its independent native token-scan reference. Those are historical
model validations attached to the measured artifact; full model benchmarks
were not rerun merely to checkpoint and push this workspace.

Previously recorded small-file GPU source-rdata loading failed in
`runtime/source_rdata.py::_RegisteredFile` with `cuMemHostRegister_v2` status 1,
before C++ permutation; neighboring CPU checks passed 29 tests. That loader
failure remains unresolved. The specific operator causing native token-scan
versus batched-prefill BF16 differences also remains unisolated. AMD/NVIDIA
hardware execution and the newly merged AMD native compiler were not validated
in this Iluvatar environment.
