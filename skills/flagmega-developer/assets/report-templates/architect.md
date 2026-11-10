---
run_id: <run>
agent_id: <actual instance>
assignment_id: <assignment>
role: Architect
domain: <Joint | Graph | Vector | Distribution | Kernel>
attempt_index: <reserved attempt index or null>
iteration_index: <qualified index or null>
iteration_qualification: <complete numerical receipt and allocation, or explicit not-qualified reason>
baseline_id: <baseline>
trial_id: <trial or pending>
revision_id: <revision or pending>
status: <plan_ready | revise_plan | insufficient_evidence>
---

# Architecture handoff

## Conclusion and hypothesis

<Lead direction, whole-workload cost and why this experiment is next.>

## Frozen inputs and bottleneck

| Input / evidence | Identity / path | Observed fact and scope |
| --- | --- | --- |
| <baseline, source/IR/artifact, profile/measurement> | <hash, link> | <fact; unknowns explicit> |

## Conditional benefit and falsification

| Model / units | Assumptions / calibrated versus theoretical ceilings | Expected range / bound | Counter-cost | Refuting observation |
| --- | --- | --- | --- | --- |
| <roofline / critical path / Amdahl> | <actual arithmetic/memory path> | <estimate, not measured> | <cost> | <check> |

## Expert baselines and SOTA target

| Adopted implementation / revision / license | Reused or ported symbols / mapping / memory strategy | Required numerical / ABI adaptations | Matched baseline reproduction | Improvements beyond it / measured gap |
| --- | --- | --- | --- | --- |
| <implementation starting point and source pin> | <concrete components and local paths> | <adaptation diff and reason> | <receipt or pending owner/action> | <proposed changes; outperform target stays open until a strict measured win> |

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <affected domain/ID> | <primary-source URL, commit/symbol and search date> | <fixed scope/unit/direction; actual, reported_only or bound> | <predeclared measured/theoretical target and baseline gap> | <proof/receipt or planned check; owner/action or specific not_applicable reason> |

## Domain contracts and integration

| Contract | Before → proposed after | Legality / numerical / ownership proof | Required evidence |
| --- | --- | --- | --- |
| <affected domain> | <change> | <proof or pending item> | <path / check> |

<When affected: partition/role scope, warp allocation, waits/lifetimes and
demonstrated versus intended overlap; joint reports link all expert inputs.>

| Expert check ID | Inspected checkpoint / values | Baseline finding / proposed contract / status | Evidence / next validation |
| --- | --- | --- | --- |
| <assigned G/V/D/K check IDs> | <exact scope> | <actual finding; unknown or not_applicable with reason> | <pin/link or planned check; Vector type inventory when affected> |

## Implementation and invalidation

| Lead / collaborators | Change / public implementation path | Earliest resume stage | Invalidated plans / ABI / resources |
| --- | --- | --- | --- |
| <experts> | <normal registration/pass/override> | <stage> | <list/reason or unchanged proof> |

## Validation and uncertainty

<Causal regressions, device cases, independent review domains, agreed scenarios/
metrics, rejection rule and remaining unknowns.>

## Kernel microprofile

<For Kernel work: baseline ncu report/export and actual WG/hardware trace/SVG
identities; capture scope and module/group map; measured overlap/waits/bubbles,
causal hypothesis and falsifying observation; proposed capture changes and
overhead/clock limits. State ncu/WG/hardware-interval/analysis gates separately,
including pending owner/action or paused_ncu_error. Follow the microprofile
handoff reference; otherwise give a specific not_applicable reason.>

## Next handoff

<Optimizer task and hypothesis.md revision; exact inputs and completion gate.>
