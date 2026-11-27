# Operator/kernel profiling, recorder and swimlanes

Operator microanalysis uses the kernel swimlane: kernel profiler owns captures/analysis/swimlanes; kernel optimizer owns production
recorder integration. reviewer independently audits evidence and ordinary acceptance;
orchestrator owns leases/failure state. See [kernel profiler](agents/kernel-profiler.md).

## 1. Workflow

1. Declare the smallest hypothesis-testing layer/tile/window and complete cooperative
   warp/dependency scope. Pin workload/native/library/GPU-model/launch/provider semantics. Apply
   [GPU model equivalence](../SKILL.md#gpu-model-equivalence); record the actual
   card for execution provenance rather than requiring one physical card.
2. Before expensive model preparation/formal capture, run a small real NCU smoke
   capture under the device lease. Before tuning, inspect baseline NCU and actual
   WG/module evidence; recapture each evaluated revision's affected scope in the same cycle.
3. Choose compatible recorder/provider, validate a small complete capture and
   ordinary/disabled/minimal/full controls; normalize reviewed intervals for the renderer.
4. Inspect actual SVGs and independently audit raw-to-chart scope/timebase/attribution,
   waits/overlap/bubbles and perturbation. Retain separate capture/analysis gate states.

## 2. Checklist

<a id="capture-gates"></a>

1. Retain NCU version/command/sections/metrics, source/workload/launch, replay/cache/
   clock settings, actual exit, raw .ncu-rep and inspected usable export. Empty,
   wrong-kernel or unusable metrics fail even at exit0. Installed tool behavior
   follows the [profiler guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
2. Any NCU error sets paused_ncu_error: stop new optimization/build/benchmark
   dispatch, safely quiesce owned jobs and retain failures. Narrow initial-capture/
   harness repairs are allowed; only a successful inspected capture closes the gate.
   Instrumented timing/another profiler cannot bypass an explicit NCU requirement.
   Other platforms need an explicitly agreed equivalent native provider/tool.
3. Reuse user-authorized sudo access; otherwise explain actual error and the precise
   command requiring authorization. Keep credentials in private stdin/channels,
   never argv/logs/persistent files/whole-file sourcing; do not silently alter drivers/
   sudoers. Each retry has a new capture ID, causal repair and stopping condition.
4. Store REVISION_DIR/profiles/kernel/CAPTURE_ID reports/raw/manifest/analysis plus
   revision SVG/data/renderer/review pins; graph profiler publishes reviewed latest SVGs under root figures/.
   NCU, WG coverage, hardware activity and analysis
   are separate gates; missing intervals remain unavailable/pending, not idle or accepted.

<a id="raw-trace-contract"></a>

1. Preserve invocation/rank/device/SM/CTA, actual WG/warp membership, role/layer/
   tile/iteration, mask/observer census, expected/observed events, capacity/drop/
   overflow/unmatched pairs, operation/module/correlation and buffer/epoch/barrier.
   Include all cooperating groups, executed divergence and drain endpoints;
   leader events describe that observer, not every warp/device.
2. Retain raw integer timestamps, source/resolution/calibration/offset uncertainty
   and replay/pass/window IDs. Per-SM clocks cannot align by unrelated origin
   subtraction; separate launches/passes are distinct executions, not one shared axis.
3. observed_activity means provider-established activity at declared scope/resolution;
   inflight_envelope is issue/commit-to-completion/wait-return lifetime, not continuous
   engine busy. sampled/inferred/unknown remain visibly distinct. Coarse bins require
   verified boundaries and user agreement, never invented per-WG engine endpoints.
4. Name actual CUDA/SIMT/matrix/TMA/cp.async/load-store paths and module-to-WG
   attribution. Unused modules need proof; relevant missing activity stays unavailable.
   Interval unions/intersections use same-execution clocks and explicit window/A/B
   denominators. Report fill/steady/drain, starvation/backpressure/reuse/sync waits,
   producer lead/lag and critical-path impact; envelopes establish possible concurrency only.

<a id="recorder-method"></a>

1. Standalone IKET is additional diagnosis alongside separate NCU. Same-invocation
   WG/NCU hardware analysis needs NCU-compatible instrumentation. Document installed
   runtime/provider compatibility; run-iket sizing/collection conflicts with NCU/CUPTI.
   SASS-patching emitters are not standalone recorders by copying timer/write sequences.
2. Stable event IDs carry explicit payload width/signedness across API/raw ABI/decoder;
   wider-than-32-bit CuTe literals need explicit types. Lossless hi/lo halves retain
   observer/correlation/epoch/mask binding. Sentinel/unexecuted work emits no fake range;
   LIFO ranges nest, concurrent/cross-iteration work uses tokens. Separate issue and completion.
3. Timestamp before bookkeeping; hoist static metadata/owned stream bases, evaluate
   bounded disjoint cursors/vector stores/compact IDs. Version ABI/decoder and preserve
   dynamic identities/drop detection; no silent wrap. Prove timer encoding/granularity/
   wrap/epoch reconstruction or keep full integer time. Inspect actual PTX/SASS for
   timer/store/address/spill placement; volatile/clobbers are not sync/completion proof.
4. Compare device-kernel durations/event counts/resources on matched implementation/
   workload/launch/reset/boundary: tracing removed, recording disabled, minimal actual
   recording, full events, full with/without NCU. Pin distinct binaries/registers/shared/
   spills/residency. Host time or combined setup deltas are not pure event/profiler cost;
   do not subtract mean overhead from individual intervals or add barriers creating the schedule.

<a id="renderer"></a>

Copy [example.json](../assets/kernel-swimlane/example.json), replace synthetic
intervals with reviewed evidence and use the standard-library renderer:

```sh
python skills/flagmega-developer/scripts/render_kernel_swimlane.py \
  skills/flagmega-developer/assets/kernel-swimlane/example.json \
  --output /tmp/kernel-swimlane.svg --summary /tmp/kernel-swimlane-summary.json
```

| Normalized object | Contract |
| --- | --- |
| Document/panel | schema_version=1, explicit synthetic, integer window_ns, title/panels; panel label/invocation/scope/timebase/lanes |
| Timebase/lane | source/positive integer resolution_ns; alignment synthetic/validated_common/wg_only; unique lane ID, wg/hardware kind, scope/intervals; status captured/unavailable/not_used/not_applicable |
| Interval | Panel-unique ID, integer relative start_ns/end_ns, label/kind/evidence; retain provenance/tile/buffer/correlation |
| Phase/dependency/bubble | Explicit fill/steady/drain boundaries; arrows reference interval IDs; positive wait spans classify starvation/backpressure/buffer_reuse/synchronization/unknown with reason/evidence and critical_path true/false/null |

1. Fixed palette: index yellow, transfer light blue, tensor blue, SIMT orange,
   wait vermilion, sync purple, store green, unknown gray. Observed solid,
   envelope faint/long dashed, sampled hatch, inferred short dashed, unknown crosshatch.
   Width defaults1500, range900–4000; [preview](../assets/kernel-swimlane/example.svg) is synthetic.
2. Preserve overlaps in subtracks without inventing warps, points as vertical marks
   and true short durations. Hardware-on-WG axis needs validated_common; independent
   invocations compare scale, not concurrency. Subtract validated integer origins
   before conversion; cropping never moves completion to the crop boundary.
3. Summary uses interval unions by evidence, avoiding double counting. Observed
   index/transfer/tensor/SIMT/store direct_work_union_ns is null when absent;
   uncovered time is not idle/utilization. Missing/unused lanes have null work.
   Bubble classification/criticality is analyst-supplied, not automatic diagnosis.
4. Retain complete raw scope/census/clock/drop/manifest and SVG/data/renderer pins.
   Provide overview/indexed details; inspect actual rendering, gallery selection,
   failed loads/stale exports and any bitmap selection binding. Respect SVG-only.
   CPU parser/storage preflight and merged_samples=0 do not prove full capture.
   Reuse only verified bindings/fixtures/epochs/resets/warmup and allowed sessions;
   retain fresh-process requirements and never double-count control receipts as samples.

## 3. Index

1. [Kernel expert](agents/kernel-expert.md), [execution gates](agents/orchestrator.md#execution-gates),
   [ordinary verification](agents/reviewer.md), [workspace layout](workspace-layout.md).
2. [IKET guide, CUTLASS 954c8d7](https://github.com/NVIDIA/cutlass/blob/954c8d7be60d730cc3f51e8f1ffe9850dce054d5/media/docs/pythonDSL/guides/iket_profiling.rst)
   and [official frontend](https://github.com/NVIDIA/cutlass/blob/954c8d7be60d730cc3f51e8f1ffe9850dce054d5/python/CuTeDSL/cutlass/cute/experimental/iket/iket.py).
3. [Independent CUDA emitter 75d44af](https://github.com/revollllt/iket-cuda/blob/75d44af866b3b7147381ee0e6454bc9efa4aa94e/include/iket/iket.cuh): pin adopted revision and inspect per-file license.
4. [Renderer](../scripts/render_kernel_swimlane.py), [JSON template](../assets/kernel-swimlane/example.json),
   [agent report](../assets/report-templates/agent.md).

## 4. Lessons

1. Follow wait → buffer/epoch dependency → upstream issue/readiness → producer
   lateness → critical-path impact, with counters/instructions and falsification.
2. Issue is not completion; wait return can lag readiness. Missing clocks/controls
   block trace interpretation, not unrelated ordinary checks; actual NCU error pause still applies.
3. Historical build/sparse_mla/tle_flashmla_vs_flashmla_trace.svg supplies stacked
   light panels/relative scale/WG-Producer style, not a runtime dependency. Its
   valid-stage timing is instrumentation-inflated; overloaded CSV role/issue events
   and shared initial Q waits cannot become hardware busy intervals or repeated tile waits.
4. IKET's 32ns encoding is implementation-specific; smaller records do not prove
   lower perturbation. Changed native/recorder/decoder identities reopen affected
   clocks/census/resources/reviews, not unrelated completed acceptance.
