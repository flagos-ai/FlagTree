---
run_id: "<run>"
milestone_id: "<mNNN or null for baseline>"
revision_id: "<rNNN or null for baseline>"
agent_name: "<canonical eight-type registry name>"
agent_id: "<actual independent context identity>"
assignment_id: "<assignment>"
phase: "<baseline | plan | implementation | assessment | measurement | analysis | verdict | reconciliation>"
plan_version: "<version or scoped not_applicable reason>"
plan_hash: "<frozen plan body SHA256>"
tracking_snapshot: "<phase sequence/path/SHA256>"
source_hash: "<source pin>"
artifact_manifest_hash: "<manifest pin>"
attempt_index: "<reserved attempt or null>"
iteration_index: "<qualified index or null>"
iteration_qualification: "<complete NUM receipt/artifact/allocation or unqualified reason>"
baseline_id: "<baseline>"
trial_id: "<trial or not_applicable reason>"
review_independence: "<actual reviewer context/scope; otherwise not_applicable>"
status: "pending"
---

# Agent handoff

## 1. Decision and scope

<Lead with actual verdict, owned scope, exact revision and unmet criteria in the
user's language. Distinguish measured facts, estimates, unavailable and not-run evidence.>

| Agent | Required decision content |
| --- | --- |
| orchestrator | Actual milestone/revision state, gain/signature gates, accepted best, next dispatch/resources and publication receipts |
| graph expert | Joint semantic/vectorize/distribution G/V/D contract, causal hypothesis, expected gain/counter-cost/falsifier |
| graph optimizer | Implemented graph/representation/distribution mechanism, integrated artifacts and resume/invalidation/self-checks |
| kernel expert | Native/pipeline/resources/lifetimes K contract, adopted baseline, bottleneck hypothesis and falsifier |
| kernel optimizer | Owned kernel/recorder implementation, resource/ABI evidence and frozen self-check readiness |
| graph profiler | Matched primary workload/operator samples, graph/history/backend figures, uncertainty and audited render provenance |
| kernel profiler | NCU/WG/hardware scope/raw/controls, swimlane analysis and separate capture/review gates |
| reviewer | Actual independence, source/NUM/state/resources/timing/figures/reproduction verdict, ledger IDs and gain/signature gate limits |

## 2. Inputs and outputs

| Item | Exact identity / durable path / changed contract |
| --- | --- |
| Task / plan / inputs | <revision plan/tracking, source/artifact/reference/profile and upstream contract pins> |
| Actual outputs | <owned code/contract/manifests/results/raw/SVG/response/handoff; unavailable needs reason> |
| Baseline adoption / target | <source/date/revision/license/reused symbols/adaptation and matched target/result/gap; not_applicable reason if appropriate> |
| Reused evidence / independence | <prior report/contract/inventory pins and unchanged lineage; actual authors/reviewer context and scope> |

## 3. Checklist and evidence

| Assigned check / shared gate | Actual inspected scope / command | Outcome / exit / reason | Evidence / pending owner-action |
| --- | --- | --- | --- |
| <O/G/V/D/GO/K/KO/GP/KP/R ID> | <CPU/source/build/device scope> | <pass/fail/not_run/unavailable/not_applicable> | <actual durable receipt/pin and next check> |

| Measurement / publication | Actual identity / result | Limits / required next action |
| --- | --- | --- |
| Every variant/scenario/round | <ledger IDs/constituents, raw samples, boundary/statistic/unit/gain/uncertainty/validity> | <primary gain verdict and missing gates> |
| Graph/operator/history/backend figures | <artifact/workload/verification/capture/raw/data/SVG/render pins> | <coverage/overhead/aggregation/independent raw-to-chart review> |
| Kernel capture/analysis | <NCU/WG/hardware/controls/raw/SVG and reviewer pins; separate gate states> | <missing evidence/owner or explicit not_applicable> |
| Reproduction | <actual tracked tests/benchmark receipt and historical redraw/local-data pins> | <unmet checks/inputs> |

<Link large inventories/manifests instead of copying them. Authors self-check;
profilers measure/capture/render; reviewer independently verifies and writes the
ledger; technical owners publish; orchestrator controls flow. Follow [agent contracts](../../references/agents/orchestrator.md#agent-registry).>

## 4. Signature and next handoff

<Write the actual agent's scoped signoff in handoff.json using its required fields;
include decision/UTC/actual context, revision/plan/source/manifest/phase snapshot
and evidence pins. Plan owner/reviewer transcribe the receipt/hash into plan/tracking.
Placeholders do not approve anything; follow [signature rules](../../references/workspace-layout.md#signatures).>

<Next recipient, exact packet, runnable command/cwd/resources, actual prerequisites,
open findings and completion condition. Retain failed/no-gain evidence; unsigned
work or numeric/build success cannot achieve a performance milestone.>
