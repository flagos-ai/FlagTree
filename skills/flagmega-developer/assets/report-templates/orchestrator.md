---
run_id: <run>
agent_id: <actual instance>
assignment_id: <assignment>
role: Orchestrator
domain: Joint
attempt_index: <reserved attempt index or null>
iteration_index: <qualified index or null>
iteration_qualification: <complete numerical receipt and allocation, or explicit not-qualified reason>
baseline_id: <baseline>
trial_id: <trial or not_applicable>
revision_id: <revision or not_applicable>
status: <continuing | complete | blocked | paused>
---

# Flow handoff

## Decision

<Current outcome, next action and reason; all unmet criteria.>

## Contract and workspace

| Item | Pinned identity / durable path |
| --- | --- |
| Task and acceptance contract | <task.yaml and identity> |
| Delivery / run / external mappings | <workspace.json> |
| Current baseline / candidate | <IDs and source/IR/artifact receipts> |
| State / overall trace / performance history | <paths, latest event/measurement IDs> |
| Progress dashboard / public snapshot | <entrypoint, snapshot time, completed/active/next work and current timing boundaries> |
| Expert coverage | <Graph/Vector/Distribution/Kernel responsibilities, assigned directories, current review/lineage and actual active assignments> |
| Attempt / numerical qualification / iteration allocation | <attempt ID; complete numerical receipt and exact artifact; assigned iteration or null reason; next counters> |

## Gates and accepted best

| Revision | Build | Independent review | Numerical / state / resources | Performance | Reproduction | Decision / evidence |
| --- | --- | --- | --- | --- | --- | --- |
| <ID> | <status> | <status> | <status> | <status, scenario> | <status> | <reason, links> |

<Accepted exact artifact or none; proposed candidates remain pending.>

## Expert baselines and SOTA target

| Adopted implementation / revision / license | Reused or ported symbols / mapping / memory strategy | Required numerical / ABI adaptations | Matched baseline reproduction | Improvements beyond it / measured gap |
| --- | --- | --- | --- | --- |
| <specialist starting point and source pin> | <implementation lineage receipt> | <approved adaptation identity> | <Verifier receipt or pending owner/action> | <consolidated gap; continue the outperform objective until a strict measured win> |

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <affected domain/ID> | <specialist baseline and research receipt> | <joint matched scope and metric> | <target, consolidated outcome and unmet gap> | <Architect/Optimizer/review/Verifier records; next owner/action> |

## Dispatch and resources

| Assignment / actual agent | Role / domain | Pinned inputs | Output path | Dependency / next action |
| --- | --- | --- | --- | --- |
| <assignment> | <role> | <identity> | <path> | <action> |

<Device/lock and active jobs/exits, active Verifier ledger writer, agreed budget.>

<Ready/running/waiting assignments; each wait's actual prerequisite and owner.
Time to first decisive NUM/timing result; time spent in build, device checks,
profiling, review and export. State what changes if supporting work dominates.
Separate provisional ordinary results from final evidence/acceptance completion.>

## Changes and unresolved work

<Completed receipts, retained failures/revisions, amendments, cleanup/export
decisions, blockers and not-run checks.>

## Kernel microprofile

<For Kernel revisions: KernelOptimizer owner, capture/device assignments and
baseline/candidate ncu/WG/hardware-interval/analysis gate states; raw-data/SVG
and independent review/export receipts. On ncu error, record paused_ncu_error,
owned-job quiescence and access/retry action; reuse already authorized sudo
access without exposing credentials. Keep missing module intervals pending.
Follow the microprofile handoff reference; otherwise give a specific
not_applicable reason.>

## Next handoff

<Recipient, concrete task, immutable inputs and completion condition.>
