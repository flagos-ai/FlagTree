# Performance iteration

Use this reference when tuning runtime performance, recovering a regression,
changing a numerical or storage contract, or preparing final measurements.
It describes decision criteria, not a fixed sequence of tuning parameters.

Kernel development also follows [Kernel microprofiling](kernel-microprofiling.md):
mandatory ncu, actual per-WG pipeline and hardware-module activity traces,
swimlane SVGs and before/after overlap analysis throughout the development loop.

## Keep the iteration loop short

Follow the [candidate prerequisite table](agent-flow.md#candidate-checks-and-acceptance).
Prioritize a runnable ordinary artifact, resource/ABI checks, complete NUM and
the first decisive timing over finishing diagnostic infrastructure. Independent
reviews, fixture preparation and profile analysis can run in parallel. Their
actual execution prerequisites still apply, and final acceptance waits for all
required evidence. An active ncu failure always retains its global pause.

Track two separate costs: time to validate the next hypothesis, and execution
time of the resulting artifact. Profile compiler passes, constant materialization,
packing, and device compilation independently. A faster resume is not a faster
kernel, and a warm readonly-data cache is not an end-to-end compile-time gain.

Reuse content-addressed readonly data when its source and transformation recipe
are unchanged. Avoid repeated import, distribution search, or packing when the
hypothesis only changes a later stage. Keep the previous best immutable and
record the input checkpoint, compiler/target identity, local extension/override
source, changed decisions, acceptance results, and raw timing samples per trial.

Choose the earliest checkpoint whose incoming contracts remain valid:

| Change | Resume and invalidate |
| --- | --- |
| IR semantics, rounding, types, or effects | Resume before the affected transformation; recompute downstream analysis and decisions. |
| Distribution choices | Reopen the distribution proposal and solve coupled choices without accidentally pinning unrelated defaults. |
| Candidate parameters, capabilities, workspace, or ABI | Resume before candidate proposal generation; regenerate selection and dependent memory/synchronization plans. |
| Contract-compatible kernel source or renderer override | A valid later TIR checkpoint may be reusable; regenerate source and device code with the override loaded and new source identity. |
| Executable compiler/resource flags with unchanged IR and ABI | Reuse verified final TIR and unchanged readonly transformations; rebuild through the normal artifact API, recording the executed flags/compiler and new library identity. Recheck actual resource admission and complete numerical correctness. |

Rebuild only the stage the hypothesis touches. An artifact holds three
independent things -- the readonly weight data, the generated source, and the
compiled device library -- and a template or kernel-source edit invalidates only
the last two. Re-running the whole artifact write re-packs gigabytes of weights
for nothing; measured on the 28-layer Qwen3-1.7B BI-V150 C++ artifact:

| Action after a template edit | Cost |
| --- | --- |
| Re-render source into the existing artifact | ~4 s |
| Recompile the device library | ~18 s |
| Re-pack readonly weights (not needed) | the rest of ~28 s, plus 3.9 GB of disk per trial |

For a new unfrozen, unreviewed and unmeasured revision, use the normal compiler's
artifact build/update path to render source, rebuild the library and refresh the
`codegen.source` / `codegen.library` hashes and `nbytes` in the manifest. Verify
that the resulting full artifact loads; a library-only build is insufficient.
Do not hand-patch emitted code or pair an old manifest with a new library.
Reuse unchanged readonly sections without copying
model-sized weights. In-place rendering applies to mutable build outputs;
never overwrite a frozen baseline or candidate under review or measurement.
Skipping the hash refresh fails the load with
`Artifact section 'codegen.source' size does not match its manifest`, which is
the loader correctly refusing a stale pairing rather than a bug to work around.

Two practical consequences. Each full re-write costs disk as well as time --
repeatedly copying whole artifacts per variant filled a 438 GB scratch volume
twice in one session and started failing builds with `No space left on device`.
For closed trials after measurement, redundant large build outputs may be
removed once trial records, generated source/final IR, provenance, reproduction inputs and raw
evidence are preserved. Retain the active baseline, pending review/validation/
reproduction/export candidates, accepted deliverable and shared readonly data
still referenced by retained artifacts. And when a template
edit needs several
attempts, the per-attempt cost decides whether to think or to try: at ~22 s a
sweep is cheap, at ~30 minutes it is not, which is reason enough to find the
in-place path before starting the sweep.

Validate index arithmetic off-device before rebuilding anything. A packed-layout
walk, a permuted copy, or a loop restructure is pure integer algebra, so a host
model that enumerates what each thread multiplies and compares the multiset per
output against the reference form settles correctness in under a second. That
oracle found a real defect -- a chunk-base shortcut that was wrong for 217 of
512 bases because the chunk can cross a row boundary -- after three device
rebuilds had failed to localize it, and it also showed that two earlier device
measurements had been attributed to the wrong cause.

Do not infer compatibility from a retained candidate ID. A source override that
changes buffer access, scratch requirements, execution roles, or transfer
lifetimes belongs in the contract-changing case. An unchanged semantic IR hash
does not establish executable identity after a template or kernel edit.

Ordinary resume preserves decisions; reselection is an explicit action. Do not
erase hashes, thaw frozen state, or patch old proposals to force acceptance.
Stopping at a boundary already reached must not execute later stages. When
changing resume infrastructure, test both the stage name and output-stage alias,
extension reloading, and repeated resume without duplicate transformations.

## Check three contracts together

### Compiler time: reuse proofs rather than weaken the workload

For a complete compilation target, measure imported IR loading, all compiler
stages, source emission and fresh readonly-data publication together. Record
checkpoint byte auditing and GPU JIT separately. An audited source identity is
different from an rdata cache; state which one was reused.

First look for graph-size work repeated per local edit or per immutable stage:
index rebuilding, inverse-user scans, rule dispatch on impossible roots,
type-tree hashing/encoding, recipe lookup and repeated ABI construction. Retain
unchanged object identities and reuse bounded analysis/encoding working sets.
Invalidate proofs on changed local inputs, inference/verifier signatures and
ambient contracts; always retain global topology, function, recipe and TIR
checks. Derived caches must not enter IR equality, serialization or semantic
visitor traversal. Preserve canonical content digests exactly.

Factor solver relations only when complete legality, cost and shared-resource
dependencies are equivalent. Keep original candidate/hard-selection decisions
and objective priorities, and test extraction against the actual chosen types.
Try static capacity, affine owner-origin and uniform interval proofs before
symbolic tail algebra; undecided cases still need the full ownership proof.
Keep fail-before/pass-after reproducers and before/after end-to-end timings.

**Numerical semantics.** Separate storage dtype, computation dtype, reduction
ordering, intermediate rounding, and result dtype. Removing a materialized Cast
is not by itself permission to remove its rounding: under an exact numerical
contract, a fusion must execute the required conversion inside the kernel.
Under an explicitly relaxed contract, simplify redundant conversion chains at
the earliest semantic stage instead of preserving them in an epilogue. Do not
silently apply that relaxation to other acceptance policies. Represent an
ordered conversion chain explicitly when its rounding remains required.
Do not move it across a collective, shared use, or effectful boundary without
a proof. Compare against the actual configured source execution path, not only
an algebraic formula or a differently compiled evaluator.

**Logical distribution.** Logical coordinates, packed lanes, local shard
coordinates, and scalar storage offsets are different domains. Preserve units
when vectorizing or scalarizing views and communication adapters; include every
lane component rather than assuming one component per axis. Check valid leaf
layouts for logical inputs instead of treating replication as the only option.
Keep necessary resharding and partial-reduction edges explicit.

**Physical storage and calls.** Equal tensor or distributed types do not imply
equal buffer ABIs. Check dtype, shape, strides, offsets, owner mapping, backing
storage, alias group, and lifetime. If a temporary alias escapes as a function
result, its whole MemSpan group must satisfy the public ABI. In-place reuse is
a candidate subject to physical compatibility, not an unconditional promise.
Specialization keys must cover the contracts checked by argument binding,
including parent backing storage, while identical contracts should still reuse
one implementation.

When adding an op attribute or TIR contract, audit its complete path: definition,
inference/evaluation/effects, handwritten builders and pattern helpers,
Python dump/resume, vectorization, distribution, fusion, lowering, bufferization,
and all applicable kernel variants. Preserve existing default semantics and
test that older checkpoints without the new attribute still load correctly.
Fix the layer that owns the missing contract, not a downstream verifier symptom.

## Optimize the whole execution boundary

Inspect generated IR/TIR and source to confirm that the intended transformation
actually happened. Count materializations, repeated conversions, communication,
publication barriers, descriptor construction, and duplicated function bodies,
not just arithmetic operations. Check scratch, registers, shared memory, spills,
and occupancy when relevant to the chosen backend.

Use existing implementations when they fit, or define new TIR, write a kernel,
or override an existing implementation when that better expresses the hypothesis.
Evaluate the resulting producer/consumer schedule, not only an isolated op.
Less local work can cost more globally by adding boundary restoration,
communication, synchronization, or callee specialization. Asynchronous transport,
replication, larger tiles, and more unrolling are hypotheses, not universal wins.

Preserve reusable function/codegen bodies for repeated computation. Place call
and pipeline boundaries according to actual reuse and synchronization needs;
do not automatically fragment every operation into an independent region or
duplicate implementations per invocation. Only hoist expressions after proving
argument invariance and absence of mutable-state dependencies or side effects.

## Count the work items before tuning any knob

Compute how much of the block has work before reaching for a depth, width, or
stage count. For a per-thread GEMV walk that is `work_items /
block_threads`, where `work_items` is the payload extent divided by the
per-thread chunk. A site at 25% or 75% residency is a defect that no amount of
knob tuning reaches, and one division finds it.

When the payload walk underfills the block, cooperate along the *long*
reduction axis rather than shrinking the parallel axis: give each output
element a group of adjacent threads that each own an interleaved slice of the
reduction, which restores residency while keeping each thread's read wide. Put
the slice index on the fastest-varying thread axis so the group is a contiguous
aligned lane run and folds with one shuffle instead of shared memory, and
interleave rather than block the slices so the concurrently active window stays
cache-resident.

Verify in the generated assembly that a concurrency knob changes concurrency.
An unrolled issue-and-consume body can leave the scheduler draining each stage
before starting the next, so the nominal depth is not the in-flight depth; the
knob then measures as noise and the real limiter is elsewhere. After any
bottleneck moves, re-sweep the knobs rather than trusting an earlier optimum --
a flat curve is evidence that the knob is no longer the constraint.

Convert per-family times into achieved bandwidth (GB/s) before ranking them.
An outlier can sit mid-table by milliseconds while being far off the
others' bandwidth, which is the form the defect is visible in.

When a measured-negative path is disabled in code, record *why* it lost, not
only that it did. "The wide load cannot pay for the lost residency" names the
real cause and makes the path recoverable by splitting a different axis;
"measured slower" invites repeating the same experiment.

## Build correctness evidence at the right level

Before using an evaluator as the numerical oracle for encoded weights or a
fused operation, verify that it implements the actual storage encoding and
rounding boundaries. A valid inferred BF16 result does not prove that evaluation
decodes an IQ4/Q4 byte matrix. A fused FP32 projection epilogue may also differ
from composing two BF16-returning primitive evaluators. Use small independent
format/rounding fixtures, then actual device checks; repair the owning evaluator
and neighboring encoded paths rather than accepting a misleading oracle.

For a defect, retain a minimal reproducer and identify file, function/pass,
and triggering condition. Demonstrate a real failing-before assertion or
execution, then pass-after with the causal fix. An invalid fixture, import
failure, or unavailable device is not evidence of the claimed compiler bug.
Preserve the intended optimization while fixing its cause; if its premise is
invalid, document the needed redesign instead of hiding a fallback.

For benchmark wrappers or recipe entry points, check the exact child command
and bootstrap in a fresh process before a long run. A CPU-only help or identity
probe must resolve the same helper imports; the actual build must select the
intended recipe installer. Direct module imports and mocked subprocess success
do not test that entry path. Record setup or precompilation separately from
execution, with the actual arguments and exit statuses.

For timed kernels, check the outputs and guards of the exact compiled
instantiations selected by the timing dispatch. A diagnostic specialization
with extra stores or instrumentation does not establish correctness of the
timed specialization. Capture its output before another launch or reset
overwrites the sink. Save each completed sample batch outside the measured
interval before later checks can fail; retain partial timings with their
failure gates, and mark them usable only after the actual timed path passes.

Use focused op/rule/pass/TIR and device-kernel tests, including negative cases:
shared or exported intermediates, effects, mixed dtypes, layout/vector variants,
multiple owners, empty or tail regions, alias escape, and repeated calls as
applicable. Test the actually selected implementation and adjacent variants
that share its contract. CPU suites, device execution, and skips are separate
results. Assert meaningful graph dependencies or ABI properties rather than
incidental node positions or pass counts unless those are the tested contract.

For stateful execution, run independent multi-step requests with realistic
state transitions and boundary crossings. Zero state or fixed inputs may be
useful performance probes but cannot establish state-update correctness.
If acceptance requires exact greedy sequences, compare complete independently
generated sequences; teacher-forced checks and approximate logits are diagnostic
evidence, not substitutes for the requested acceptance rule.

When exact generation fails, retain the first mismatch and actual/reference
top candidates, logits and margins before discarding the execution state.
Identify whether the complete input history is still equal at that step;
later cross-history logits cannot explain the first divergence. Establish the
current same-profile compiler baseline on the same independent workload before
attributing the failure to an optimization or revising precision. Isolate the
changed mechanisms and reproduce affected operations on identical saved inputs
and encoded rows; a family-level failure alone does not prove a decoder,
epilogue or reduction-order defect. Record storage dtype, intermediate rounding,
accumulation order and each state leaf separately. A justified profile change
requires a named task revision, fresh baseline and separate comparison cohort,
while preserving prior failures and the user's acceptance constraints.

Keep the reference from becoming part of candidate execution. Prefer isolated
state; if same-history diagnostics must share storage, snapshot and restore
reference writes before invoking the candidate. Separately run acceptance with
diagnostic reference forwards disabled, so a reference write cannot hide a
missing candidate write.

## Measure, attribute, and publish

Match the measurement boundary before comparing numbers: compiler wall time,
device kernel time, captured graph replay, model forward with or without a
sampler, and serving request latency are distinct. Include required sampling,
copies, state handoff, scheduling, and output processing in the claimed boundary.
Report excluded warmup, initialization, or cache-flush work explicitly.
Cache policy and graph mode must match the intended comparison.

Measure the real reference runtime with its configured optimizations enabled.
For serving integration, preserve ownership of scheduling, state allocation,
and sampling according to the agreed ABI. Validate zero-copy shape/stride/offset
contracts, or include conversion costs. Do not compare optimized graph execution
against an eager reference without labeling that separate experiment.

Run comparable variants sequentially on the same device, rotate order, repeat
enough to assess variation, and retain raw samples and per-round summaries.
Keep correctness diagnostics outside timing, avoid competing jobs on the measured
device, and do not rebuild native libraries or mutate artifacts during a run.
Check GPU competition before launch and throughout each measured child. A task
lease or file lock only coordinates participants that honor it. Retain timestamped
process queries and establish ownership from the actual child and its descendants;
do not allow a process merely because its executable has the same name. Record
the monitor's extent and any unavailable process identity. A later competition
snapshot does not establish overlap with an earlier completed interval, and an
empty prelaunch snapshot does not establish isolation throughout the child.
For host-wall timing, also keep compilation outside the measured execution and
record CPU compilation checks before launch and throughout the child. Report
CPU and GPU monitor results and their actual scope separately from an aggregate
guard verdict: a CPU competition failure does not make clear GPU observations
fail. A disabled or unavailable monitor is not evidence of an uncontended interval.
Keep raw samples from interrupted or contaminated attempts, but exclude affected
intervals from performance acceptance and attribution.

Probe ncu access with a small leased launch before tuning or collecting an
expensive model profile. Any ncu error or unusable report pauses optimization:
preserve the error, request sudo access if it has not already been authorized
and made available, and retry within that authorization. Resume only after a
successful inspected capture; persistent errors keep the flow paused. Follow
the [failure and credential rules](kernel-microprofiling.md#mandatory-capture-and-failure-gate).
Timing, generated-code inspection and instrumentation supplement ncu and cannot
replace it. Likewise, ncu aggregates alone cannot establish WG or hardware-module
timelines. Distinguish measured activity, in-flight envelopes and estimates.

For a grid-barrier megakernel, verify the actual cooperative-launch admission,
grid size and resident-block limits needed by its synchronization contract.
Requested register limits or launch bounds are not measured residency. Inspect
the new executable's registers and spills, then query device occupancy and
check that its complete grid fits the available resident blocks. A register cap
can increase spill/reload work; measure the resulting whole-workload effect.
Use the same resource settings on both sides when attributing a gain to a
separate layout change, and record requested settings apart from actual use.
Contention can also invalidate that contract; do not infer that every concurrent
launch necessarily deadlocks or that a hang proves contention. Serialize runs on
one device or give each run its own, and discard measurements with observed GPU
overlap. A test suite that executes kernels also counts as a GPU job.

Verify that a reference artifact was built from the same module as the variant
before comparing. An artifact left over from a different profile can still load
and run, so the comparison fails silently and in whichever direction its own
configuration happens to favor. Re-derive the baseline from the same source and
confirm it reproduces its historical numbers.
Report all requested scenarios, medians and tails, including regressions and
noise. A favorable sample or recovery to a historical range does not establish
a strict win in every scenario; do not change acceptance after seeing results.

Separate current-compiler baseline gains from workload-local agent gains.
Rebuild the baseline after shared fixes. A historical artifact with a different
numerical contract can be a labeled performance reference, not the correctness
baseline. Retain intermediate comparisons when they show which choices helped;
do not assume every additional optimization improves performance.

Choose a control that isolates the proposed mechanism. For a load-lowering or
kernel change, keep placement, grid, register cap, math and other recipes equal,
and compare an explicit off/on variant built through the same current compiler.
A new-core 32-CTA control versus a new-core 96-CTA candidate measures placement;
it does not establish the benefit of a load change shared by both. When several
factors change, name each comparison and its intended varying factor. Link
historical timings as context, without pooling them or treating separate
cohorts as a causal comparison. Numerical qualification and iteration numbering
remain independent of whether the measured mechanism improves performance.

Verifier records every performance attempt under its reserved attempt ID in the
append-only [Performance history](performance-history.md), including baseline,
reference, screening, formal measurement, rejected candidates and reproduction.
An iteration is assigned only after the complete agreed numerical checks pass;
failed/incomplete numerical attempts do not increment it. A numerical pass with
a performance regression still counts. Save samples and per-round statistics
before deciding the next attempt.
Failures retain their status/reason, not a fabricated value. At handoff render
the performance-versus-iteration history for numerically qualified revisions,
preserving regressions and acceptance status. Retain every other attempt in the
public table and optional separate attempt-axis figure, with diagnostic/invalid
scope and reasons.

Render this as one cumulative figure, with workload curves or compact panels
spanning qualified iterations. Baseline/source version changes alone do not
create a separate panel per iteration. After every execution-based verification,
also save a measured per-operator latency SVG and its pinned raw profile, outside
acceptance timing. Follow [operator profiling requirements](performance-history.md#operator-latency-after-verification)
for attribution, instrumentation overhead, overlap and unavailable evidence.

Before publication, rebuild from the original input plus saved scripts and
overrides, not only the final edited checkpoint. Recheck independent acceptance,
artifact verification, and provenance linking compiler inputs, target/catalog,
override source, generated kernel hash, and measured results. Do not combine
different implementations into one variant merely because their IDs match.
Where deterministic generation is expected, compare the fresh source with the
measured source; investigate mismatches rather than relabeling an artifact.

Generate formal accepted-comparison summaries and SVGs from accepted raw
measurements. Diagnostic/regression plots may also be exported from pinned
raw data with explicit failed or unaccepted status; they do not establish
candidate acceptance and require the same provenance and sample checks. Follow
[Workspace layout and maintenance](workspace-layout.md) for trial figures,
public source data, export receipts and the overall optimization trace. Export
only the validated source/artifact after those checks, and verify that public
reproduction does not import private experiment files. Keep rejected trials,
agent handoffs and raw evidence in the task-local archive. The public
`optimization-trace.md` summarizes meaningful outcomes, including failures;
the readme explains reproduction, incremental optimizations and final results.
