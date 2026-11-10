# Kernel microprofiling and hardware swimlanes

Read this flow for kernel development and performance changes, including tile,
warp/role mapping, producer/consumer pipelines, transfers and synchronization.
Whole-model timelines and per-operator latency bars do not satisfy this flow.

For instrumentation design and its performance evaluation, read
[Low-overhead kernel tracing from IKET](kernel-trace-method.md). Apply that
method before introducing or replacing a recorder; preserve the capture,
all-group coverage and hardware-evidence requirements below.

## Mandatory capture and failure gate

On NVIDIA targets, **Nsight Compute (`ncu`) is mandatory**. Run a small diagnostic
capture on the assigned device before tuning, then collect the relevant kernel's
counters for the baseline and each evaluated changed implementation. Pin the
native source/library, workload, launch, device, tool version, command, selected
sections/metrics, replay/cache/clock settings, exit status and raw `.ncu-rep`.
Verify that the report contains the intended launch and usable required metrics;
a zero exit with empty, wrong-kernel or unusable output is not a successful gate.
Choose supported metrics for the target and bottleneck; a single full-model
aggregate or an earlier revision's report cannot discharge this requirement.

Any ncu failure, including `ERR_NVGPUCTRPERM`, missing tools, unsupported counters,
capture errors or unusable reports, activates the following flow:

1. Preserve the actual command, error and failed artifacts, with secrets removed.
   Orchestrator records `paused_ncu_error`, the affected revision and next owner.
   Stop new optimization/build/benchmark dispatch and safely quiesce owned jobs;
   do not terminate unrelated work. No continued tuning using timing, static
   inspection, instrumentation or an alternative profiler as a bypass.
2. Request sudo access from the user if it has not already been authorized and
   made available. Explain the actual failure and the precise profiling command
   requiring access. Reuse existing authorization and credentials without asking
   again. While access is missing, optimization stays paused.
3. Use the authorized access only for the necessary profiling/diagnostic command.
   If the user specifies a local credential source, such as the `pass` entry in
   `.vscode/.env`, read that entry only when needed and pass it privately to sudo
   through stdin or an equivalent protected mechanism. Never print it, place it
   in command arguments, source the whole file as shell code, or copy it into
   reports, logs, traces, committed files or persisted environment snapshots.
   Do not generalize one user's credential location to other workspaces.
4. Diagnose the failure within that authorization. Before another capture,
   record the evidence-based repair or bounded diagnostic purpose, expected
   observation and stopping condition. A fresh attempt gets a new capture ID;
   preserve failed attempts and never loop retries merely to obtain a pass.
   Sudo is not a cure for every error; do not silently change driver settings
   or sudoers.
   Keep optimization paused if the command still fails or permissions cannot be
   obtained in the current environment. Only a successful, inspected ncu capture
   closes this gate. Record its receipt before resuming; honor any separate user
   pause or scope restriction that remains in effect.

Preparation needed to obtain the initial capture, and narrowly scoped repairs
to the failing profiling harness, are allowed. They do not authorize another
optimization candidate or performance sweep while this gate is failed.

On another hardware platform, record its actual execution groups and hardware
modules and use an explicitly agreed native counter/trace profiler with the
same evidence and failure gates. If the task expressly requires ncu and the
target cannot run it, pause for the user's tool/target decision; never silently
waive ncu or claim it measured a non-NVIDIA device.
The ncu fields in companion checklists/templates inherit this platform rule;
record the expressly agreed native tool when that exception applies.

## Kernel development loop and ownership

1. **Map and capture the baseline.** KernelArchitect identifies the bottleneck,
   actual lowered roles, physical warp membership, transfers and dependencies.
   KernelOptimizer builds a correct native reproducer and obtains successful ncu
   evidence plus the trace and swimlanes below before choosing a tuning change.
   KernelArchitect specifies each module's capture provider/event semantics and
   whether it can observe activity, completion or only an envelope. Validate
   the proposed capture on a small scope first; unavailable hardware observations
   need an explicit follow-up or user scope decision, not a promised timeline.
   For a megakernel, retain the relevant outer layer/tile pipeline and resource
   context; explain limits of a standalone leaf reproducer.
2. **Explain the pipeline.** Quantify producer/consumer progress, waits, buffer
   reuse, fill/steady-state/drain and hardware-module overlap. Tie the causal
   hypothesis to counters, emitted instructions and a falsifying observation.
3. **Change and recapture.** Freeze one candidate, check correctness and actual
   resources, then repeat ncu and the affected WG/module traces under matched
   conditions. Changed scheduling, code, launch or instrumentation invalidates
   the corresponding earlier evidence. Compare baseline/candidate gaps, stalls
   and overlap; retain rejected candidates and unsuccessful captures.
4. **Independently review.** An independent KernelArchitect checks the raw
   reports/traces, clock and attribution validity, coverage, chart regeneration
   and overlap calculations. Missing trace/module evidence remains an open gate.
   Source-only review may admit ordinary checks or a diagnostic build with their
   execution prerequisites met, but cannot pass microprofile completion or bypass
   `paused_ncu_error`.
5. **Verify and hand off.** Verifier records microprofile gate status alongside
   independent numerical, resource and uninstrumented performance outcomes.
   Orchestrator carries missing artifacts into the next dispatch and exports the
   reviewed figures/data. A full-model speedup does not close missing microprofile
   evidence. No acceptance/completion claim with required evidence absent.

KernelOptimizer owns instrumentation, collection and analysis; KernelArchitect
owns the causal interpretation and independent review in a separate context.
An assigned device owner may execute captures for them under the device lease.
This does not defer the Kernel agents' responsibility to final model validation.
Use the existing roles; no new standing profiling agent is required. When
instrumentation is substantial, Orchestrator may assign a bounded capture/helper
task within the available delegation and device budget. KernelOptimizer still
owns its integration, evidence quality and completed microprofile handoff.

### Scheduling and capture scope

Follow [candidate checks and acceptance](agent-flow.md#candidate-checks-and-acceptance).
Ordinary correctness/resource checks and initial screening may precede completed
candidate profile analysis when their prerequisites hold and no ncu failure is
active. Schedule mandatory candidate ncu and WG/module captures in that same
evaluation cycle. Keep required evidence pending for an unsafe/failed candidate
with the actual reason; never claim it captured successfully or silently waive it.
Source-only review and an ordinary timing win cannot close this flow.

Start with the smallest predeclared layer/tile/window that tests the hypothesis
and includes every cooperating WG and dependency. Reuse the verified recorder,
decoder and renderer when their source, ABI, clock and coverage contracts hold.
Expand scope for an unresolved question, changed pipeline or missing coverage,
not to regenerate every historical view. Scoped capture still requires all WGs
and relevant hardware modules in that scope; it is not all-device evidence.

Recorder perturbation controls are prerequisites for interpreting instrumentation
effects and accepting its pipeline conclusions. They are not prerequisites for
the ordinary artifact's independent correctness/timing unless that artifact shares
the defective mechanism. Repair and independently review changed instrumentation
with its own source identity. Preserve previously declared conditions through a
reasoned plan amendment; no retry-to-fit, fitted offsets or relaxed thresholds.

### Capture preparation and reuse

KernelOptimizer supplies the declared scopes, commands and expected outputs
together using a reviewed, parameterized runner. A scope, output path or capture
ID change need not create another copy of the harness or another full harness
audit. Check the changed configuration's coverage, capacity, resources and source
lineage against its admission; changed recorder, lifecycle, clock or isolation
logic requires affected review. When parameterizing scopes or control modes,
exercise the actual host binding path with a lightweight CPU context before
expensive model preparation. Check admission-to-graph scope/ranges/offsets for
each supported mode, including disabled recording: zero device trace arguments
do not prove that preallocated graph metadata uses the requested scope. Preserve
valid controls after a localized failure and repeat only invalidated evidence.
Include the post-capture inspection commands in CPU preflight: exercise their
original CLI parsers with the complete proposed arguments, including required
output paths, without loading models or report/trace payloads. Checking a script
hash or `--help` alone does not prove its actual invocation is valid. Reuse this
check while the parser and arguments remain unchanged.
Inspect each actual NCU report promptly before advancing the device
queue so an unusable capture activates the failure gate immediately.

Before a large capture or batch, compare free space on each output filesystem
with the planned trace allocation, state dumps, profiler/replay intermediates
and analysis outputs, retaining a stated margin for logs and concurrent writers.
Estimate from the declared scope and existing comparable captures; do not run
another capture just to estimate storage. Restore insufficient space before
expensive preparation or taking the device lease, using the
[retention rules](workspace-layout.md#maintenance-and-retention).
After a storage interruption, preserve incomplete outputs and the failure,
reuse independently qualified unaffected controls, and give only the failed
capture a fresh output namespace and measurement ID. Recheck capacity before
retrying; never turn a partial output into a successful capture receipt.

Choose preparation reuse before collection, using measured stage costs:

- Reuse frozen build artifacts, fixtures and decoded facts whose source,
  workload, toolchain and relevant contracts remain valid. Consumers pin the
  existing integrity/audit receipt; a new owner alone does not require another
  complete payload read. Missing or changed identity invalidates that reuse.
- Reuse a qualified observation for multiple evidence roles when the original
  contract permits it. For example, a full-recording control may also supply
  the before-profile control if its artifact, scope, mode, state, ordering and
  required freshness match. Bind both roles to that one receipt; never count
  it as two independent samples. Retain the new profiled and after-profile
  captures, required independent processes/resets and drift/perturbation checks.
  Record elapsed separation and intervening work; do not infer an uninterrupted
  pair or isolated profiler cost. Reuse does not cross scope boundaries or waive
  an explicit fresh-capture requirement.
- Reuse one loaded/prepared session across scopes or controls only when the
  profiler lifecycle and measurement contract allow it. Establish identical
  required starting state, reset/warmup behavior, trace buffers/epochs, graph
  bindings and isolation for each invocation. Preserve separate invocation,
  pass and capture identities; session reuse never permits joining WG events
  and hardware samples from different launches onto one measured axis.
- When fresh processes or independent preparations are required, retain them.
  Reuse allowed CPU artifacts and batch command preparation instead. A proposed
  lifecycle change needs a reasoned Architect amendment and affected review
  before execution; do not change resets, sampling limits or acceptance scope
  merely to save setup time. Failure to establish reuse leaves the existing
  capture path available, without opening an unbounded harness redesign.

After the protected device window, independent evidence review and compact
figure preparation can proceed in parallel on frozen inputs. Publication waits
for the applicable review; missing exports do not occupy the device lease.

Audit raw records and derived overlap once per exact input/decoder/analysis
identity, then reuse the independently verified compact summary for rendering.
Keep full raw data and coverage checks. Render a lightweight overview and the
detail views needed for the decision; a wider gallery can be generated in parallel
with subsequent CPU work. Check changed SVGs actually render. Unchanged figures
do not need a new full-gallery rendering/review pass.

## Raw trace contract

Collect actual timestamped events for **every WG in the declared captured scope**.
Define kernel invocation, rank/device, SM/CTA, layer/tile/iteration and time window
before capture; enumerate all group IDs and warp membership. Distinguish hardware
warp groups, compiler specialization roles and analyst-defined groups. Uniform
kernels still show their real groups; do not invent dedicated producer roles.
Represent inactive or unexecuted groups explicitly. Sampled CTAs/windows must
have a stated selection rule and cannot establish all-device coverage.
The scope must include all cooperating groups and dependencies for the pipeline
being analyzed; selecting only the producer cannot establish consumer overlap.

The raw event records and manifest must support these fields:

| Evidence | Required content |
| --- | --- |
| Identity and scope | Capture/run/revision/source/library identities; device/rank, kernel invocation, SM/CTA, WG and participating warp IDs; role, layer/tile/iteration. |
| Event meaning | Stage, event kind, start/end or point timestamp; instruction/operation and hardware module; transaction/correlation ID; buffer slot, epoch/phase and barrier ID when applicable. |
| Pipeline boundaries | Producer acquire/load/prefetch/issue/commit, consumer wait/compute, barrier arrival/completion observation, release/reuse and idle/wait intervals as implemented. Mark missing observations explicitly. |
| Timebase | Clock source, frequency/resolution, units, calibration/offset uncertainty and synchronization scope; raw timestamps retained. |
| Coverage and validity | Expected/observed groups and events, active masks/leader sampling scope, capacity, dropped/overflowed records, unmatched events, capture window and replay/pass identity. |
| Measurement effects | Instrumentation source/configuration, diagnostic and ordinary build identities, actual register/shared/spill/residency changes, and measured overhead or explicit pending result. |

NCU counter collection alone is not evidence of per-WG timestamps. Add reviewed
opt-in device instrumentation or supported hardware tracing to obtain the
required events; retain ncu as the mandatory counter source. Keep instrumentation
in reproducible source/overrides and the normal build path. Use bounded buffers
and event ordering that preserve the kernel's ownership/synchronization rules;
do not add a barrier that creates the overlap being claimed. A leader timestamp
describes the leader's observation, not automatically every warp's progress.

Choose the collection backend before implementing the recorder. NVIDIA's
`run-iket` uses driver profiling resources and cannot run simultaneously with
NCU or other CUPTI profilers in the documented version. A standalone IKET run
can supply additional WG diagnosis, with NCU collected separately. A required
same-invocation WG/NCU hardware timeline instead needs an NCU-compatible
instrumentation path; do not combine two profiler launches into one timeline.
Check compatibility against the installed versions and record the choice.

Use a validated common timebase within each simultaneous trace. Per-SM cycle
clocks cannot be aligned across SMs by subtracting unrelated launch timestamps.
Record calibration and uncertainty before making cross-SM/device claims. NCU
passes/replays and separate launches are distinct executions: counters from them
may contextualize a trace, but their intervals must not be stitched together to
simulate concurrency. Collect uninstrumented timing separately; assess replay,
cache/clock controls and tracing's effect on the observed schedule.

## Required WG and hardware-module swimlanes

Produce an actual-data **SVG swimlane chart** for each captured baseline/candidate
scope. Interactive Perfetto/HTML views may supplement the SVG. Use one common
time axis with two linked panels, or equivalent nested lanes:

- **WG lanes:** one lane for every group in scope, with per-warp sublanes when
  their behavior differs. Show producer/consumer stages, compute, transfers,
  synchronization, waits and idle gaps; label tile/iteration and buffer epochs.
- **Hardware-module lanes:** show the working intervals of each relevant module,
  e.g. CUDA Core/SIMT execution, Tensor Core/MMA/WGMMA, TMA or other actual
  asynchronous transfer paths, load/store units, and the target's analogous
  compute/copy/matrix modules. Distinguish cp.async from TMA and identify the
  actual mechanism. State hardware scope (SM/subpartition/device), instruction
  path and attribution granularity. Show concurrent work without collapsing it
  into a serial lane; link intervals to issuing WG, operation, tile and buffer
  via correlation IDs or annotated dependencies.

For dense traces, provide a lightweight overview and indexed per-CTA/group,
stage or tile detail SVGs. Keep the complete raw trace; label each view's selected
scope and any display aggregation. Verify that the delivered SVGs actually
render, rather than publishing one document with every recorded interval.

For an interactive gallery, verify that the visible selection, rendered data
and exported filename always identify the same view. Clear and disable previous
exports when a selection starts or its load/decode fails. Bind asynchronous PNG
decode/encoding to the captured selection, SVG and filename, and discard work
whose selection changed before completion. Exercise actual browser failures and
selection changes during both decode and encoding; a successful first render
does not cover these cases. Reuse the latest qualified viewer repair separately
from immutable historical data and SVGs, so copying an older gallery cannot
silently restore a known export defect.

Module labels must follow the emitted implementation and target capabilities.
When the user accepts sampled hardware activity, measured counter activity bins
can satisfy the hardware-lane requirement at their verified sampling resolution.
Record that agreement, bin boundaries, actual interval/aggregation and evidence
type. This does not establish exact engine start/end, continuous busy time within
a bin, or per-WG attribution. Preserve the same-execution/timebase requirements
before correlating these bins with WG events; separate unaligned panels.
An unused/absent module is `not_used`/`not_applicable`, justified by code and
capture. A relevant module with no measured trace is `unavailable`, not idle or
zero utilization. Retain the lane/coverage entry and an explicit next capture
action; do not mark the requested hardware-interval deliverable complete.

Use a visible legend and machine-readable evidence class for each interval:

| Class | What it supports |
| --- | --- |
| `observed_activity` | Direct trace/instrumentation observations that establish this module's activity at the stated resolution; document the event semantics and limitations. |
| `inflight_envelope` | Issue/commit through observed completion or wait return. This bounds an asynchronous operation's lifetime; the engine may be idle during part of it. |
| `sampled` / `inferred` | Sampled activity or a modeled interval, with resolution, method and uncertainty; visually separate from direct observations. |
| `unknown` | Missing coverage/boundaries; no invented duration or utilization. |

Issuing an instruction is not its completion, and a wait's return can lag
completion. WG compute spans do not by themselves measure continuous CUDA Core
activity. Counter percentages, instruction counts, PC samples and source order
must not be expanded into fabricated engine start/end bars. In-flight envelopes
and inferred rows are useful supplementary evidence but **do not fulfill a
requirement for measured hardware working intervals**. Keep that gate pending
until supported activity evidence is collected or the user changes the scope.

## Overlap analysis and acceptance artifacts

Report fill/steady-state/drain separately. For same-execution, comparable-clock
interval sets A and B, compute overlap from their unions: `duration(A ∩ B)`.
State whether a reported fraction divides by A's active time, B's active time
or the observation window. Separate active-work overlap, waits, bubbles,
starvation/backpressure and buffer-reuse stalls; preserve per-WG/pair and
per-module results rather than only an average. Quantify producer lead/lag and
relate dependencies to observed gaps. Label envelope intersection as overlap
of in-flight envelopes (possible concurrency), never as confirmed concurrent
execution or a measured hardware-busy overlap ratio. Delayed wait observations
can make envelopes intersect even when the hardware operations ran serially.

Require, for the exact baseline and evaluated candidate:

- Raw `.ncu-rep`, a usable counter export, exact collection commands/settings
  and success/error receipts.
- Raw WG and module event trace (JSONL/CSV or a documented trace format), group
  topology, clocks, coverage/drop counts, correlation and provenance manifest.
- WG **and hardware-module** swimlane SVG, its source dataset and reproducible
  renderer/command; optional interactive trace viewer output.
- Quantitative overlap/wait/bubble analysis, ncu-supported bottleneck, before/after
  comparison, overhead/resource effects and falsification outcome.
- Independent raw-to-chart/calculation review and explicit complete/pending
  status for ncu, WG coverage, hardware intervals and analysis.

Store captures under the exact revision's `profiles/kernel/<capture-id>/` and
figures as `figures/kernel-swimlane-<capture-id>.svg` within that revision.
Follow the configured `workspace.json.artifact_layout` for reviewed copies: when
the user reserves delivery-level `figures/` for final performance charts, retain
WG/hardware and operator plots under `.local/figures/` or `.local/results/` and
link them from the dashboard. Local placement does not remove profiling,
coverage, perturbation or independent-review requirements. Shareable exports include the
relevant SVGs, self-contained redacted event data, manifest, renderer and review
receipt; large native reports can remain in the recorded private archive. The
public renderer must work without `.local/` unless the user explicitly chose
local figure data; document that dependency. Audit interval matching, clock
comparability, module attribution, coverage and computed statistics from raw
records before publication. Synthetic diagrams can explain a proposed design,
but must be labeled and cannot substitute for measured traces or close gates.

See [agent reports](agent-report-templates.md#kernel-microprofile-handoff),
[Kernel checks K6/K7/K10](expert-checklists.md#kernel) and
[profiling history](performance-history.md#kernel-microprofiling-artifacts).

Tool behavior and permission diagnostics should be checked against the installed
version and NVIDIA's [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [performance-counter access guidance](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
