# Performance iteration

Use this reference when tuning runtime performance, recovering a regression,
changing a numerical or storage contract, or preparing final measurements.
It describes decision criteria, not a fixed sequence of tuning parameters.

## Keep the iteration loop short

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

**Numerical semantics.** Separate storage dtype, computation dtype, reduction
ordering, intermediate rounding, and result dtype. Removing a materialized Cast
is not permission to remove its rounding: a fusion can execute the conversion
inside a kernel. Represent an ordered conversion chain explicitly when needed.
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

## Build correctness evidence at the right level

For a defect, retain a minimal reproducer and identify file, function/pass,
and triggering condition. Demonstrate a real failing-before assertion or
execution, then pass-after with the causal fix. An invalid fixture, import
failure, or unavailable device is not evidence of the claimed compiler bug.
Preserve the intended optimization while fixing its cause; if its premise is
invalid, document the needed redesign instead of hiding a fallback.

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
Report all requested scenarios, medians and tails, including regressions and
noise. A favorable sample or recovery to a historical range does not establish
a strict win in every scenario; do not change acceptance after seeing results.

Separate current-compiler baseline gains from workload-local agent gains.
Rebuild the baseline after shared fixes. A historical artifact with a different
numerical contract can be a labeled performance reference, not the correctness
baseline. Retain intermediate comparisons when they show which choices helped;
do not assume every additional optimization improves performance.

Before publication, rebuild from the original input plus saved scripts and
overrides, not only the final edited checkpoint. Recheck independent acceptance,
artifact verification, and provenance linking compiler inputs, target/catalog,
override source, generated kernel hash, and measured results. Do not combine
different implementations into one variant merely because their IDs match.
Where deterministic generation is expected, compare the fresh source with the
measured source; investigate mismatches rather than relabeling an artifact.

Generate summaries and SVGs from accepted raw measurements. Export only the
validated source/artifact after those checks, and verify that public reproduction
does not import private experiment files. Keep rejected trials and raw evidence
in the task-local archive; the user-facing document should explain reproduction,
the agent's incremental optimizations, and final results rather than the work log.
