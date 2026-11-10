# Low-overhead kernel tracing from IKET

Use this method when introducing or changing in-kernel instrumentation,
especially for warp specialization, producer/consumer pipelines and async
transfers. It extracts the event-design and profiling method from NVIDIA IKET;
it does not require replacing the compiler/backend or installing IKET for every
task. The [mandatory microprofiling flow](kernel-microprofiling.md) still governs
NCU, scope, hardware lanes, failure handling and independent acceptance.

## Select the collection path first

| Required result | Collection path | Evidence boundary |
| --- | --- | --- |
| Additional in-kernel WG diagnosis with supported IKET kernels | Compile IKET events, use `run-iket`, retain its raw output and Perfetto trace; collect mandatory NCU in a separate run | Separate executions: compare mechanisms, but do not align their intervals or infer concurrent hardware activity |
| WG stages and NCU hardware samples from the same invocation | Use opt-in device instrumentation compatible with NCU; borrow applicable IKET event/encoding ideas without requiring its runtime injector | Verify actual invocation, timebase, replay/pass identity and full coverage before producing a shared axis |

The documented `run-iket` runtime conflicts with NCU, Nsight Systems and other
CUPTI profilers. It also uses sizing and collection passes. Do not nest these
profilers, use IKET to bypass an NCU failure, or relabel different executions as
one capture. Recheck compatibility for the installed versions. A standalone
IKET run can complete a separate diagnostic substep; it does not complete this
workflow's mandatory NCU gate or a requested same-invocation joint analysis.

## Design events around the pipeline

Start with a coarse event map of each real role, then add detail where it can
answer the bottleneck question. A useful map is producer acquire → issue/commit
→ consumer wait → compute → release/reuse. Include loop fill and drain, buffer
slot/phase changes, and final outstanding completions. KernelOptimizer owns the
map and implementation; KernelArchitect checks the question and semantics.

Use stable event names or IDs for repeated phases. Put tile/iteration or
transaction identity in a small typed payload instead of generating a new name
per iteration. IKET's documented encoding can widen above 30 unique names;
treat that as an IKET implementation detail, not a universal event limit. Do
not remove required observations merely to stay below it.

Specify payload width and signedness explicitly and verify the installed API,
raw ABI and decoder preserve them. In the documented CuTe DSL API, plain
Python literals use 32-bit payloads; use explicit `cutlass.Int64` or
`cutlass.Float64` for corresponding 64-bit literals. Token-range endpoints
must use matching payload types. Never silently narrow a required value. If a
backend needs a hi32/lo32 representation, bind both halves to the same
observer, correlation ID, epoch and selected active lane, and verify lossless
round-trip decoding. Include the extra records in capacity and overhead counts.

Use point markers for instantaneous observations, stack ranges only for LIFO
nesting, and token/explicit-ID ranges for overlapping or cross-iteration
operations. A sentinel for an operation not yet started must not generate a
fake start/end event. Close every executed dynamic range, including the drain
iteration and divergent paths that actually run.

Place warp-level events on warp-uniform paths where possible. Record active
mask and leader/payload selection semantics. IKET payloads select an active
lane; this does not preserve every participating thread's value. A pipeline
with distinct thread-level producers inside one warp needs explicit observer
IDs and lossless recording of each required producer. Do not collapse them
into one warp event or invent extra physical warp groups.

For asynchronous operations, record issue/commit and completion observation
separately. An endpoint immediately after TMA/cp.async/MMA issue measures the
issue path. A wait-return endpoint observes readiness and may lag actual
completion. Their envelope is not a hardware-engine busy interval. Show actual
module activity using the required hardware provider or accepted NCU samples.

Coarse-to-fine instrumentation controls event density, not coverage authority:
retain every cooperating WG and dependency in the declared capture scope.
Diagnostic subsets need explicit scope labels. Display selection/aggregation
does not authorize dropping required raw events.

## Keep the recorder's hot path small

Inspect the existing recorder before assuming that inline assembly makes it
cheap. Count state loads/stores, address calculations, counter updates and
record bytes at each event, in addition to timer instructions. Use the
following design choices when they fit the actual ABI and ownership model:

- **Observe before bookkeeping.** Capture the intended timestamp before trace
  counter updates and record stores that would delay that observation. Keep
  event placement relative to the model operation explicit. An entry marker
  after recorder initialization is not the hardware kernel start.
- **Hoist stable metadata.** Put source identity, event dictionaries and known
  launch geometry in a manifest/header. Cache a stream's buffer base and cursor
  in registers when their lifetimes and resource cost permit it. Avoid reading
  configuration and updating a global count for every record when a final
  commit can preserve capacity, ordering and failure detection.
- **Use owned, bounded streams.** Prefer disjoint per-observer append regions
  over a contended global append counter. Check capacity and preserve explicit
  overflow/drop and incomplete-stream evidence. Do not wrap and silently
  overwrite earlier events. Size from the lowered schedule/census where
  possible; for dynamic workloads retain a verified bound or sizing evidence.
- **Encode only what is needed on the hot path.** Evaluate compact event IDs,
  small payloads and aligned/vector stores. Version the raw ABI and decoder.
  Reconstruct static fields only from pinned, unambiguous metadata; do not
  discard dynamic branch/mask/SM/buffer/transaction information needed by the
  trace contract. Verify lossless decoding against actual records and census.
- **Make ordinary builds clean.** Prefer compile-time removal of tracing for
  the uninstrumented performance artifact. Verify actual generated code;
  runtime-disabled instrumentation can still consume registers/instructions.

These are adaptation candidates, not claims about IKET's private runtime.
The public CUDA C++ IKET emitter demonstrates early timer reads, compact IDs,
small shared-memory writes and `pmevent` anchors. Those sites depend on
`run-iket` SASS patching and metadata: its scratch address, event-bit encoding
and marker instructions are not a standalone general-purpose trace buffer.
Do not copy that sequence into an NCU recorder and assume it records events.

If using a shortened timer or embedding IDs into timestamp bits, prove the
target's timer granularity, reserved bits, wrap interval, full-width anchor and
epoch reconstruction across all streams. Preserve integer timestamps until
subtracting a validated origin. IKET documents 32 ns event granularity; that
does not establish another recorder's clock or NCU sampling resolution.
Keep full-width time or a separate ID field when the encoding proof is absent.

## Verify generated code and execution

Inspect PTX **and actual SASS/native instructions** for the changed capture
sites. Locate timer reads, configuration/state accesses, spills and final
record stores. Prove that stores use the intended captured timestamp and that
compiler motion did not move the observation across the operation of interest.
`asm volatile` and a compiler memory clobber are not device synchronization or
proof that an asynchronous operation completed. Do not add synchronization that
creates the pipeline behavior being measured.

Query actual registers, shared/local memory, spills, launch geometry and
residency. Shared-memory staging or long-lived cursor registers can reduce
global traffic while increasing occupancy cost. Compare the generated ordinary
and diagnostic implementations, and pass the agreed numerical/state checks.

Validate a small complete capture before scaling up: expected groups/events,
pairing, masks/ownership, slot/epoch reuse, capacity/drop counts, ordering,
clock calibration and the raw-to-decoded mapping. Preserve failed captures.
For a shared NCU axis also inspect the actual pass/group, usable timestamp
vectors and full workload coverage; `merged_samples=0` alone does not prove an
untruncated trace. Declare alignment criteria and endpoint semantics before
capture. An entry observation delayed by bookkeeping calls for a causal
recorder/observation repair, not retries until one passes or a fitted offset.

Changed recorder code or ABI gets a new native identity. Revalidate affected
resources, NUM, clocks, census and joined captures; do not carry a calibration
or acceptance receipt across an unverified identity change.

## Measure perturbation before interpreting trace improvements

These controls establish the limits of instrumented pipeline conclusions.
Ordinary uninstrumented correctness and performance checks can proceed under
the [candidate prerequisites](agent-flow.md#candidate-checks-and-acceptance)
while controls are prepared. Final microprofile acceptance still requires the
applicable controls; an ncu failure continues to pause optimization. Reuse a
verified recorder when its contracts hold instead of redesigning tracing for
each kernel candidate.

Separate compiler effects, per-event cost and profiler overhead. Use matched
device-kernel timing boundaries and preserve raw samples and executed event
counts. Useful controls are:

| Control | What to compare |
| --- | --- |
| Ordinary artifact with tracing removed | Correctness and actual uninstrumented kernel/model performance |
| Trace-capable artifact with recording disabled | Code-generation/resource cost of the instrumentation path |
| Minimal actual instrumentation | Combined minimal recorder/profiler cost; retain real event sites and required backend setup |
| Intended full event set | Incremental event cost, resource effects and the pipeline used for diagnosis |

For `run-iket`, follow its documented minimal-event versus full-event procedure
using trace-reported GPU kernel durations under the same profiling setup. The
tool must recognize the minimally instrumented kernel. Treat the duration
difference as an estimate of incremental event overhead; state the event/warp
count denominator for any amortized figure. An enabled but unprofiled IKET
binary is a separate compiler sanity check, not that overhead experiment.
Minimal profiled minus minimal unprofiled timing includes recorder/injection
changes as well as profiling. Report this combined effect; do not call it pure
profiler cost without a control that isolates that component.

For our NCU-compatible recorder, keep the underlying kernel implementation,
workload, launch and device timing boundary matched, and pin every diagnostic
native variant. Use one native binary for recording-mode toggles when supported;
otherwise report compilation/resource differences as part of the measured
delta. Separately quantify the effect of NCU on the full recorder. Host wall
time includes loading, injection, export and decoding and does not isolate
per-event overhead. Keep
end-to-end uninstrumented timing for the model's performance objective.

Do not subtract a measured average overhead from individual trace intervals.
Compare source/SASS/resource changes and measured distributions to decide
whether the new recorder perturbs the pipeline less. A smaller record or fewer
instructions alone does not establish lower execution overhead.

## Handoff and source provenance

Include the event/role map, selected collection path and profiler compatibility;
recorder/native/decoder identities; upstream symbols or mechanisms actually
adopted; format/capacity/wrap proofs; inspected SASS locations; NUM, raw census
and timebase results; perturbation controls; and independent review. Emit the
required actual WG/module SVGs with compact regeneration data, explicit scope,
resolution, evidence classes and links to full raw evidence. Use an overview
and indexed details for dense data; verify the actual SVGs render.

Keep ownership in the existing KernelOptimizer/KernelArchitect/Verifier flow.
Do not create a standing profiling-agent role or weaken the NCU/sudo gate.
Distinguish literal code reuse, ABI compatibility and independently implemented
ideas. Inspect per-file licensing before copying source; the CUTLASS CuTe DSL
subtree has its own terms, and a public repository is not by itself a code-copy
license. This reference summarizes methods and includes no copied recorder.

Primary sources inspected for this method:

- [NVIDIA IKET guide, CUTLASS revision 954c8d7](https://github.com/NVIDIA/cutlass/blob/954c8d7be60d730cc3f51e8f1ffe9850dce054d5/media/docs/pythonDSL/guides/iket_profiling.rst): event placement, warp semantics, compatibility, sizing and overhead experiments.
- [Official IKET frontend at the same revision](https://github.com/NVIDIA/cutlass/blob/954c8d7be60d730cc3f51e8f1ffe9850dce054d5/python/CuTeDSL/cutlass/cute/experimental/iket/iket.py): marker/range/token API; this file emits IR operations, not the runtime's recorder assembly.
- [CUDA C++ IKET emitter, revision 75d44af](https://github.com/revollllt/iket-cuda/blob/75d44af866b3b7147381ee0e6454bc9efa4aa94e/include/iket/iket.cuh): an author's IKET 0.5 ABI emitter, with inline PTX and `run-iket` dependencies; not an official portable recorder API.

Pin the versions actually used in a task and recheck their supported ABI,
target and compatibility. These revisions are provenance, not a requirement to
install old versions or use an undocumented ABI.
