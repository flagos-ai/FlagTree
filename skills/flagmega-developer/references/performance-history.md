# Iteration performance history

Verifier records every performance attempt so the final report can show how
performance changed over iterations. The ledger supplements the overall trace
and does not replace raw evidence or acceptance.

## IDs, writer and timing

**Only a candidate that passes the agreed numerical validation counts as an
optimization iteration.** Orchestrator reserves a monotonically increasing
`attempt_index` before each candidate revision enters evaluation. Initially its
`iteration_index` is null with a qualification reason. Build failures, numerical
failures, not-run checks and partial diagnostics keep their attempt IDs and all
evidence, but do not consume an iteration number.

Verifier checks the exact frozen artifact against the complete numerical scope
fixed in `task.yaml`, including every requested scenario and applicable state,
sequence and eager/graph/reset correctness checks. Self-checks, source review,
mock results or a shortened screening pass do not qualify. This rule does not
add new accuracy thresholds or require matching tokens beyond the agreed
generation contract.

After Verifier records that complete numerical pass, Orchestrator assigns the
next `iteration_index` (1, 2, ...) and pins the qualification receipt and artifact
identity. A numerically correct candidate counts even if performance regresses
or another acceptance gate fails. Numerical qualification is not final
acceptance. Scenarios, rounds and repeat measurements of the unchanged revision
share one iteration; they do not each increment the counter. A changed revision
gets a new attempt and must qualify independently. Reproduction of the same
candidate does not create another iteration.

A numerically validated baseline may use index 0 within its comparison cohort;
it is not an optimization iteration. An unvalidated baseline has a null index.
New core baselines create explicit cohorts, not a reset of the run's candidate
iteration counter. Reference measurements do not qualify on behalf of a
candidate; link them to a qualified comparison iteration when applicable.

Verifier is the single writer of
`WORK_DIR/metrics/performance-history.jsonl`. Reserve a unique `measurement_id`
before each attempt. Serialized Verifier assignments continue the ledger after
checking its last IDs and active writer. Other agents send their measurements
to Verifier; retain both `measured_by` and `recorded_by` assignments.

Save raw results under the exact baseline/revision, then append a record before
reporting its outcome or advancing the iteration. Record every variant,
scenario and round; an aggregate references its constituent measurement IDs.

## Record schema

| Field | Required content |
| --- | --- |
| Identity | Schema version, unique measurement ID, UTC start/end, run/attempt/iteration/baseline/trial/revision IDs, measuring and recording assignments. `iteration_index` is null until complete numerical qualification; identify any baseline/reference association separately. |
| Comparison | Cohort ID, variant (`baseline`, `candidate`, `reference`), scenario/shapes/steps, kind (`screening`, `formal`, `profiling`, `reproduction`), primary metric/unit/direction and timing boundary. |
| Execution | Checkpoint/reference/compiler/source/recipe/IR/manifest/generated-source/library identities, device UUID, environment and graph/cache/configuration. Link immutable receipts instead of huge copied manifests. |
| Samples | Raw path/hash, measured/excluded warmup counts, round/repeat/order, sample count, mean/median/p95/max and throughput where relevant; percentile and aggregation method. |
| Gates | Current review identity, numerical/resource/performance/reproduction status, actual exit, measurement validity, candidate decision and reason; separate CPU/GPU monitor results, actual scope and aggregate guard verdict; complete numerical scope, qualification receipt and Orchestrator iteration-allocation receipt, or explicit reason no iteration is assigned. |
| Evidence | Raw results, stdout/stderr, telemetry/profile and validation paths/hashes; estimates remain separate. |

Failed, skipped, unavailable and not-run attempts still get records with their
status/reason and null unavailable metrics. Partial actual samples can remain
explicitly partial diagnostics. A fast measured result may still be invalid;
retain its values and failure instead of promoting it to accepted performance.

The ledger is append-only. Later verdict changes use a `gate_update` event
referencing measurement IDs and new evidence. Value corrections reference the
original record and explain the error; preserve original samples. Orchestrator
links record IDs and decisions into the overall trace.

Dependencies on an active ledger identify an immutable copied snapshot or a
byte prefix (size and SHA256). Verify that the live file is at least that long
and that the frozen prefix is unchanged; allow later appends. Do not require
whole-file hash equality after reserving an ID or recording a result. Exercise
the real dependency validator before and after simulated reserve/result appends
on a private fixture, with corrupted/truncated-prefix rejection controls.

Numerical qualification and iteration allocation are also append-only updates
referencing the exact revision and its measurement IDs. Do not rewrite pending
records to make it appear that an iteration was assigned before validation.
For legacy runs that reserved iteration numbers before validation, preserve
those numbers as `legacy_attempt_index`, migrate them to `attempt_index` where
appropriate, and append explicit corrections to the effective iteration field.
Keep frozen reports and raw ledger events unchanged; publish the migration
reason and provenance. Do not promote older partial passes during migration.

On adoption/resume, Verifier may backfill actual older receipts: label them
`backfilled`, preserve original measuring actor/timestamp/scope and identities,
and identify missing provenance. Do not invent iteration ordering or upgrade
historical diagnostics into formal measurements.

## Comparable cohorts and final chart

A cohort fixes core baseline, numerical/reference contract, workload/device,
graph/cache policy and timing boundary. Shared core fixes create a new cohort
and rebuilt comparison inputs; do not connect unexplained baseline changes.
Keep profiling, short screening and formal full-workload series separate.

Expert SOTA comparisons reference their pinned comparison IDs and targets from
[Expert baselines](expert-baselines.md). Record actual external-baseline and
candidate microbenchmarks under their own matched workload/cohort and boundary;
link them to the expert's outcome. Published-only numbers and theoretical bounds
remain labeled evidence outside actual sample statistics. A local operator win
does not replace whole-workload acceptance or qualify a full-model iteration.

Export compact public history in `results/<export-id>/summary.json` and render
the latest `figures/performance-vs-iteration.svg` through public `render_results.py`.
Keep only latest SVGs in `figures/`; store data, CSVs, receipts, dependencies and
historical SVG versions in `results/<export-id>/`.
This is one cumulative history figure containing all qualified iterations,
following the evolution curves in the [EAT README](https://github.com/caoshiyi/EAT).
Each workload/measurement contract has a curve spanning its iteration history;
use compact workload panels in the same SVG when needed. Do not create one
performance figure or panel per iteration. Source revisions and core baseline
versions remain point provenance and comparison-series identities; their change
alone must not split an otherwise comparable history into isolated panels.
Include attempt/iteration/measurement IDs, numerical qualification and its
receipt, cohort/variant/scenario/configuration,
sample counts/statistics, execution identity hashes, validity and latest verdict.
Public rendering must work with `.local/` unavailable.

Publish the updated export after each verification handoff, including rejected
results; do not wait for final performance acceptance. Orchestrator embeds the
latest cumulative SVG in the delivery README, beside links to its compact data
and public rendering command. The caption identifies the covered iteration
range, workloads, measurement kind, gaps and acceptance status. Preserve older
exports and update the README pointer rather than overwriting their evidence.

When storage is constrained, the public renderer may support
`summary-<export-id>.json.gz` and `performance-history-<export-id>.csv.gz`.
Compress the same complete public records; retain qualifications, regressions,
failures and missing-sample reasons. Use deterministic gzip (`mtime=0`, no
embedded filename), and record both the stored container and decompressed
payload size/SHA256. A renderer's input-file hash identifies the stored file;
it does not substitute for the payload identity or independent data review.
Verify that plain and compressed inputs produce identical validated records,
figure points and decompressed CSV bytes through the public entrypoint without
private run files. Preserve previously published exports.

For decode TPS, plot decoded token count divided by total measured seconds
(`1000 / pooled mean ms/token`), with token/s and higher-is-better labels.
Convert observed round means using the same definition; do not average the
reciprocals of individual token durations. Preserve original sample units,
qualification and gaps in the source data and render report.

The primary x-axis is the count/index of numerically qualified optimization
iterations; the y-axis is the agreed metric with units. Only qualified candidates
advance that axis, with a qualified baseline at 0 when available.
Use workload panels or explicit comparison-contract series, show repeated-run variation and
reference/baseline lines only from comparable measurements. Retain numerically
qualified regressions and candidates rejected by other gates with explicit
markers. A qualified iteration without valid performance samples has a gap and
reason, never zero latency. Failed or incomplete numerical attempts remain in
the public history table and may have a separate figure with `attempt_index` as
its clearly labeled x-axis; they must not advance or appear as qualified points
on the primary iteration curve. Keep screening/profiling values distinct from
formal performance even after numerical qualification.
A best-so-far curve may supplement the actual history but cannot replace it.

Keep a measurement's recorded numerical status and iteration assignment
distinct from its revision's current qualification. A later complete pass does
not automatically assign its iteration to earlier failed or cancelled records.
Preserve their original null index, event/time and provenance in the history
table and attempt diagnostics. If the renderer associates such a row with a
now-qualified revision, require an explicit historical association and exclude
it from qualified iteration points. Effective reassignments require actual
append-only qualification/correction records; do not invent another attempt or
drop historical rows to satisfy a rendering schema.

Verifier checks coverage, raw aggregation, boundaries/units, provenance, latest
verdicts and captions before export. Every numeric point traces to samples;
incompatible or unmeasured iterations remain discoverable in the public table.

## Kernel microprofiling artifacts

During kernel development, collect the distinct mandatory evidence in
[Kernel microprofiling](kernel-microprofiling.md) for the baseline and each
evaluated changed implementation: ncu report/export, actual per-WG pipeline and
hardware-module event trace, correlated swimlane SVG, quantitative overlap/wait
analysis and independent review. These are additional to the cumulative history
and per-operator charts; whole-model timings do not provide microscopic evidence.

Record each diagnostic capture, including failures, with its measurement ID,
exact source/library/workload, device/launch, profiler and instrumentation
settings, overhead and raw artifact identities. Keep replay/pass and simultaneous
trace identities distinct, and preserve unsuccessful capture receipts. Profiled
timing does not enter the uninstrumented acceptance series. Numerical iteration
qualification remains separate from microprofile completion.

Retain raw ncu and trace data privately under `profiles/kernel/<capture-id>/`;
export the relevant WG/hardware swimlane SVGs, self-contained event data,
clock/coverage manifest, renderer and checked analysis to the public package.
Audit raw event matching, module attribution, clock comparability and overlap
calculations before rendering/export. Missing hardware activity intervals cannot
be replaced by inferred bars or asynchronous in-flight envelopes. Track ncu,
WG coverage, hardware intervals and analysis as separate gates in the reports,
`validation.json` and trace; missing evidence prevents completion.

An ncu error activates the mandatory pause/sudo-access flow immediately. Generic
unavailable-profile reporting below does not authorize continued optimization
without ncu.

## Operator latency after verification

For every execution-based candidate verification, Verifier produces one
`operator-latency-<verification-id>.svg` showing measured time for all executed
operators. Multiple workloads may share compact panels in that SVG. Retain each
verification's operator breakdown and raw profile; update the cumulative
iteration-history figure separately.

This deliverable applies to accepted and rejected verification results. A
verification may contain several scenarios or repeats; link their measurement
IDs to its single operator SVG. Give a later verification its own ID and SVG.

A verification is a declared candidate/scenario evaluation, not each shell
command, preflight, witness or repeat within it. Record those constituent runs
and failures individually, while sharing their verification's profiling task.
Schedule profiling independently of ordinary NUM/timing under the
[candidate prerequisites](agent-flow.md#candidate-checks-and-acceptance).
Missing operator figures retain an owner and completion condition; they do not
block unrelated ordinary checks or create a new full profiling round for each
handoff. Distinct evaluated artifacts keep their own provenance and deliverables.

Use actual profiling evidence from the exact tested candidate. Record operator
or call-site identity, layer/call counts, workload/token count, device, timing
boundary/unit, raw durations and aggregation, including rank/CTA aggregation.
Define inclusive versus exclusive time and whether synchronization/communication
is included. Preserve overlap and unexecuted/missing operators; do not double
count nested calls or sum overlapping durations as end-to-end latency. Label
separately compiled instrumentation and its overhead. Collect profiling outside
acceptance timing under its own device ownership. Numerical qualification and
iteration allocation remain independent of this diagnostic measurement.

Plot measured per-operator contribution in descending order with a clear unit
and denominator. Preserve all executed operators; a grouped remainder identifies
its members and aggregation. Percentages require a valid explicit denominator.
State whether each bar represents mean time per call, accumulated time per
token, or time over the measured workload, and retain the call/token counts
needed to reproduce it. Label operators or call sites so repeated layers and
fused operations can be traced back to the profiled implementation.
A monolithic kernel duration cannot establish inner operator times. Use a
reviewed opt-in diagnostic build when instrumentation is needed; never divide
the kernel time among operators.

Save profile JSON/CSV and SVG in the revision's private `profiles/` and `figures/`.
Link their hashes, profiling measurement IDs and attribution limits from
`validation.json`, the Verifier report and optimization trace. Retain failed
verification profiles as diagnostics where valid. If a build failed, profiling
is unsafe/unavailable, or attribution is incomplete, record the exact reason and
pending work instead of fabricating durations or an actual chart. CPU-only
preparation has no runtime operator profile. Public handoff includes relevant
operator SVGs and self-contained data after provenance checks.

Missing actual profiling leaves this deliverable pending. Verifier records its
status and next owner/action in `validation.json`; Orchestrator carries the
pending work in the trace until a later profiling receipt closes it. Keep any
already established numerical qualification and measured performance verdict
separate from this figure's completion status.

The renderer must fully audit raw actual-device records in the same invocation
before drawing actual operator bars, or reject the raw input and require an
independently verified compact summary. Supplied raw-document aggregates,
coverage flags and receipt labels do not replace that audit. A compact summary
may render without private archives after its independent full-raw aggregation
receipt and public sample arrays have been checked. Validation-only execution
must not claim that it rendered an actual chart; its report states whether it
checked structure, audited raw records, or validated a verified summary.
