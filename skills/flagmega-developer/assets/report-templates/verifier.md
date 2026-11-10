---
run_id: <run>
agent_id: <actual instance>
assignment_id: <assignment>
role: Verifier
domain: Joint
attempt_index: <reserved attempt index or null>
iteration_index: <qualified index or null>
iteration_qualification: <complete numerical receipt and allocation, or explicit not-qualified reason>
baseline_id: <baseline>
trial_id: <trial or not_applicable>
revision_id: <revision or not_applicable>
independent_of_implementation: <true | false, explain below>
status: <accepted | rejected | diagnostic_only | not_run | unavailable>
---

# Verification handoff

## Verdict

<Actual outcome and every unmet criterion; distinguish screening, formal
measurement and public-package reproduction.>

## Tested contract and identities

| Item | Exact identity / configuration / path |
| --- | --- |
| Current independent review / candidate | <review/source/IR/code/library/manifest hashes> |
| Checkpoint / reference / workload | <revisions, numerical profile, shapes, generation policy> |
| Device / runtime / timing boundary | <UUID, versions, graph/cache/warmup/reset configuration> |
| Numerical qualification / iteration allocation | <complete task numerical scope, per-scenario receipts, exact artifact; Orchestrator allocation or null reason> |

## Gate results

| Gate / scenario | Required criterion | Actual result / exit | Evidence |
| --- | --- | --- | --- |
| <identity/IR/numerical/state/resource/performance/reproduction> | <fixed rule> | <pass/fail/skip/not_run> | <raw receipt> |

## Every performance measurement

| Measurement ID / attempt / iteration / cohort | Variant / scenario / kind | Samples / mean / median / p95 / throughput | Validity / decision | Raw evidence |
| --- | --- | --- | --- | --- |
| <IDs> | <baseline/candidate/reference, screening/formal/profile/reproduction> | <actual values/unit or null reason> | <gates/exit/partial/invalid> | <sample/log/telemetry hash> |

<Append all attempts to metrics/performance-history.jsonl before handoff.
Reference every round/variant/scenario ID or an aggregate with constituent IDs;
retain failed/rejected attempts and later gate_update records.>

<A complete numerical pass qualifies one revision, even with slower performance.
Partial diagnostics and failed/incomplete numerical checks do not increment the
iteration counter. Send the qualification receipt to Orchestrator for allocation;
repeat rounds/scenarios of that revision share the assigned index.>

## Analysis and limits

<Comparable per-round deltas/variation, regressions, failure category,
same-input/history limits, unavailable counters and not-run checks.
For exact-generation failure, include the first mismatch, complete-history
equality, actual/reference top logits and margins, same-profile baseline result,
storage/intermediate-rounding/accumulation/state-leaf contracts, and unresolved
causal questions or explicit unavailable evidence reasons.>

## Expert baselines and SOTA target

| Adopted implementation / revision / license | Reused or ported symbols / mapping / memory strategy | Required numerical / ABI adaptations | Matched baseline reproduction | Improvements beyond it / measured gap |
| --- | --- | --- | --- | --- |
| <adopted starting point and reviewed source pin> | <implementation lineage receipt> | <reviewed adaptation identity> | <actual matched execution and raw samples> | <candidate result and uncertainty; outperform requires a strict measured win> |

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <affected domain/ID> | <baseline identity and source/search date> | <tested contract, actual baseline/unit/direction or reported_only> | <predeclared target, candidate measurements/uncertainty, gap> | <numerical/performance ledger IDs; measured/theoretical/gap/pending status and owner> |

## Iteration figure and reproduction

<One cumulative performance-versus-iteration figure and its draft/public dataset
paths, all numerically qualified iteration coverage,
workload/cohort/kind series, status/gap annotations and provenance checks; failed
or incomplete numerical attempts use the history table or separate attempt-axis
figure, not the primary iteration curve; pending until executed.>

| Publication handoff | Checked result |
| --- | --- |
| Latest cumulative SVG / compact public data / rendering command | <Git-managed paths and actual public-only render result> |
| Delivery README embed / coverage and status caption | <README link; Orchestrator confirms update before closing handoff, or explicit pending owner/reason> |

## Operator latency figure

| Item | Actual result / identity / evidence |
| --- | --- |
| Verification / candidate / workload | <verification ID, exact artifact/review, workload and token/call counts> |
| Profiling execution | <actual command/exit, device, source/library identity; separate instrumentation and overhead if used> |
| Operator attribution | <all executed operator/call-site coverage; inclusive/exclusive, wait/overlap and rank/CTA aggregation> |
| Statistics / units | <bar statistic, time unit and denominator; operator call/token counts and reproducible aggregation> |
| Raw data / SVG | <profile JSON/CSV and operator-latency-<verification-id>.svg paths/hashes; profiling measurement IDs> |
| Result / limits | <completed only with actual SVG and checked provenance; otherwise unavailable/not_run reason, pending owner/action and validation.json status> |

## Kernel microprofile

<For Kernel revisions: check the baseline/candidate ncu report/export, raw WG
and hardware trace, correlated swimlane SVG/data/renderer and independent
analysis receipt against the tested identities. State ncu/WG/hardware-interval/
analysis gate results, overhead and uninstrumented verification separately.
Record incomplete gates and owner/actions in validation.json; model/operator
charts cannot discharge this requirement. Follow the microprofile handoff
reference; otherwise give a specific not_applicable reason.>

## Next handoff


<Orchestrator/Architect decision input, accepted artifact or repair/hypothesis,
validation.json and ledger IDs.>
