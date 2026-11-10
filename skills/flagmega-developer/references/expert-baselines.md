# Expert baselines and SOTA targets

Every affected performance expert establishes a best-known baseline and target
for each optimization revision, then accounts for whether the implementation
reaches it. This applies to Graph, Vector, Distribution and Kernel Architects
and Optimizers. Independent reviewers check comparison and implementation
evidence; Verifier owns measured acceptance.

## Use industry SOTA as the implementation starting point

Each affected expert starts from the strongest applicable industry
implementation found in the source research. Prefer direct reuse or a
traceable port of its code, algorithm, thread mapping, memory strategy and
schedule, then make the adaptations required by the task's numerical and
runtime ABI. Record the upstream revision/license, reused symbols/components,
adaptation diff and why those adaptations are necessary. When an upstream
runtime cannot be used directly, port its applicable implementation into the
normal compiler-owned path and preserve the source lineage.

Do not write the implementation from scratch when applicable upstream code is
available. Raising the implementation starting point requires concrete code
reuse or a traceable port; merely citing SOTA, borrowing its target number or
reimplementing the same idea independently does not satisfy this requirement.

Reproduce the adopted implementation under the matched contract before
attributing new gains. Use that reproduced implementation as the baseline for
further optimization. When the task requires outperforming it, keep that
objective open until a strictly better matched measured result satisfies the
task's whole-workload acceptance rule. Revisit the starting point when research finds
a stronger applicable implementation. Keep source research, implementation
reuse, local reproduction and target attainment as explicit evidence fields;
each has its own completion state.

## Find and freeze the comparison

Search current primary sources: upstream implementations, official benchmark
code and research papers with their implementation artifacts. For operators,
inspect the actual kernel and benchmark harness, not just a performance claim.
Record search date, URLs, repository/commit or paper revision, relevant symbols,
configuration, reported results and why the selected implementation is the
strongest applicable baseline among the sources inspected. Keep credible faster
alternatives and exclusion reasons visible. State which sources were searched;
do not claim a global SOTA from one convenient implementation.

Freeze the comparison contract: operator or fused-region semantics, input and
output shapes/layouts, quantization and numerical/rounding profile, state,
device/topology, runtime mode, cache/warmup/reset policy, timing boundary,
sample aggregation and metric direction. Compare like-for-like. A faster
implementation with different precision, rounding, work, communication or
timing scope is a separately labeled reference until equivalence is established;
it cannot silently replace the task's agreed correctness or acceptance baseline.

Maintain two comparisons where useful: the current correct FlagMega baseline
for causal before/after gains, and the external best-known implementation for
the SOTA target. Preserve both identities. Published numbers remain
`reported_only` until reproduced under the comparison contract. Missing or
incompatible implementations require an explicit limitation and theoretical
target; an unrun external benchmark never becomes an actual local measurement.

Save selected sources, contracts and targets in the revision's
`plans/expert-baselines.json`; save outcomes and raw/model evidence in
`results/expert-baseline-comparison.json`. Link these records from role reports
and hypothesis, implementation, review and validation records. Archive benchmark
entrypoints and configuration with the baseline; reuse readonly data and pinned
artifacts instead of duplicating model-sized files. Reuse unchanged research and
evidence after reconfirming applicability and freshness; refresh the search when
scope, hardware, contract or relevant upstream evidence changes.

## What each expert compares

| Expert | Best-known implementation baseline | Theoretical target and implementation evidence |
| --- | --- | --- |
| Graph | Equivalent model region or end-to-end implementation with strong fusion, indexing and elimination; include all fused work and state updates in the boundary. | Necessary FLOPs/bytes/materializations/launches and dependency critical path; prove the emitted fused graph removes the claimed work without moving costs or rounding elsewhere. |
| Vector | Best applicable representation, packing/layout and access strategy on the target, including scalar access where it is faster. | Actual `TensorType.dtype`/`VectorType` lanes, lane/address mapping and function ABI; byte/transaction/issue lower bounds, conversion and tail costs, and emitted loads/stores. More lanes or wider instructions alone do not establish SOTA. |
| Distribution | Best applicable placement/sharding/collective/scheduling implementation for the same participant count and topology. | Work/communication lower bounds, load balance and DAG/rank critical path; verify actual ownership, transfers, waits, residency and overlap. Scaling on different topology is not a matched baseline. |
| Kernel | Reuse or traceably port a leading compatible operator/kernel implementation, preserving source/license and adaptation evidence; reproduce it under matched shapes, layouts, numerical semantics and device, then improve it to meet the task's strict measured outperform objective. | Appropriate compute/memory/issue roofline, resource and pipeline bounds; verify actual TIR/recipe, generated instructions, tiling, occupancy, spills and synchronization implement the adopted strategy and subsequent improvements. |

Cross-domain changes use one reconciled boundary. A graph-fusion target may need
Vector, Distribution and Kernel evidence together; local speedups cannot simply
be added to estimate end-to-end benefit.

## Reach the target or record the gap

Before implementation, state the selected metric, baseline result or bound,
target and comparison tolerance/uncertainty. Preserve the task's existing
performance acceptance rule. Each expert reports the implemented mechanism and
its outcome with one of these explicit states:

- `measured_target_met`: Verifier's matching numerical and performance evidence
  satisfies the predeclared target against the strongest reproduced applicable
  baseline under the declared metric and comparison rule. For an objective to
  outperform it, require a strictly better measured result with the declared
  uncertainty rule; matching the baseline remains `gap_remaining`. Scope the
  claim to the evaluated implementations, contract and search date.
- `theoretical_target_met`: a proof establishes optimality or attainment of the
  best-known target within a stated model/search space, and the inspected
  generated implementation realizes the modeled work, mapping and schedule.
  Give assumptions, lower bound, achievable construction, counter-costs and
  falsifying observations. Device performance remains unmeasured until checked.
- `gap_remaining`: the current result or implementation falls short; quantify
  the gap when possible and give the next bottleneck and owner/action.
- `insufficient_evidence`: missing reproducible baseline, calibration, proof or
  actual result; identify the missing evidence and the next check.
- `not_applicable`: this assignment does not optimize runtime performance, such
  as profiling infrastructure, figure tooling or a correctness-only repair;
  give the specific reason.

A roofline ceiling alone is not proof that an implementation attains it.
Distinguish hardware peak, calibrated sustainable ceiling, algorithmic lower
bound and achievable schedule. Include issue limits, dependency waits,
communication, spills and mandatory conversions when they affect the claimed
bound. Search-space optimality applies only to the declared alternatives.

Each round's expert performance objective stays open until it reaches the
measured or proven theoretical target. A ready-for-review implementation may
still have a performance gap; Orchestrator routes further work or records the
unmet target instead of treating source review as SOTA achievement. Numerical
qualification still determines whether the candidate counts as an iteration;
SOTA attainment does not change that counter. Whole-workload acceptance and
final reproduction remain required even when all local experts meet targets.
When the task requires a measured performance win, theoretical attainment is
interim evidence and cannot close that objective or replace the actual matched
comparison. Preserve both the adopted operator baseline and any separately
required whole-model reference.

Use this shared table in each applicable role report. Joint reports cover all
affected domains with specialist evidence. Non-performance assignments retain
the table with a specific `not_applicable` reason.

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <Graph/Vector/Distribution/Kernel and stable ID> | <URL, commit/symbol, search date> | <identity, unit/direction, measured or reported_only value/bound> | <predeclared target, actual result or proof, remaining gap> | <Verifier IDs or model/emitted-code proof; explicit status and owner> |
