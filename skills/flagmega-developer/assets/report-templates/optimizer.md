---
run_id: <run>
agent_id: <actual instance>
assignment_id: <assignment>
role: Optimizer
domain: <Joint | Graph | Vector | Distribution | Kernel>
attempt_index: <reserved attempt index or null>
iteration_index: <qualified index or null>
iteration_qualification: <complete numerical receipt and allocation, or explicit not-qualified reason>
baseline_id: <baseline>
trial_id: <trial or not_applicable>
revision_id: <revision or not_applicable>
status: <ready_for_review | needs_repair | blocked>
---

# Implementation handoff

## Result

<Implemented behavior, self-check outcome and review readiness.>

## Input and output identities

| Item | Identity / path |
| --- | --- |
| Hypothesis / input source / complete IR | <revision, hashes, links> |
| Owned changes / public recipe or override | <source freeze, install path> |
| Final IR / generated source / library / manifest | <hashes or not_run reason> |
| Implementation record | <implementation.json> |

## Changes and domain contracts

| Change / owner | Mechanism | Preserved or changed contract | Actual emitted evidence |
| --- | --- | --- | --- |
| <change> | <implementation> | <dtype/layout/writer/effects/ABI> | <source/IR/span proof> |

| Expert check ID | Inspected checkpoint / values | Actual result / reason | Evidence |
| --- | --- | --- | --- |
| <assigned G/V/D/K check IDs> | <exact scope> | <pass/fail/not_run/not_applicable; reason> | <self-check receipt; Vector type inventory when affected> |

## Expert baselines and SOTA target

| Adopted implementation / revision / license | Reused or ported symbols / mapping / memory strategy | Required numerical / ABI adaptations | Matched baseline reproduction | Improvements beyond it / measured gap |
| --- | --- | --- | --- | --- |
| <implementation starting point and source pin> | <concrete components and local paths> | <adaptation diff and reason> | <receipt or pending owner/action> | <implemented changes; outperform target stays open until a strict measured win> |

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <affected domain/ID> | <approved baseline pin/URL/commit/date> | <matched contract/unit/direction and baseline> | <target; actual result or achievable-model/emitted-code proof; gap> | <Verifier IDs or theoretical evidence; explicit target status and owner/action> |

## Resume and plan regeneration

<Earliest stage used, replaced choices/dependent plans, unchanged-contract proof
for reused plans and deviations from the approved hypothesis.>

## Self-checks

| Check / actual command | Scope (CPU/mock/source/NVCC/GPU) | Before → after / exit | Evidence |
| --- | --- | --- | --- |
| <check> | <scope> | <actual result, skips/not_run explicit> | <log/hash> |

<Any measured screening: scenario/boundary/samples and Verifier ledger IDs;
estimates remain separate, no independent acceptance claim.>

## Risks and incomplete gates

<Defects, environment limits, unmeasured workloads and required reviews.>

## Kernel microprofile

<For Kernel work: own the baseline/candidate ncu capture receipts and report
exports, raw WG/module events and coverage/clock manifest, correlated WG and
hardware-activity swimlane SVG/data/renderer, overlap and pipeline analysis,
instrumentation identity and overhead/resources. Separate actual module activity
from in-flight envelopes. State ncu/WG/hardware-interval/analysis gates and
missing owner/actions; ncu failure pauses tuning. Follow the microprofile
handoff reference; otherwise give a specific not_applicable reason.>

## Next handoff

<Independent reviewer domains, frozen inputs and concrete review questions.>
