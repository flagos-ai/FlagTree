---
run_id: <run>
agent_id: <actual instance>
assignment_id: <assignment>
role: Reviewer
domain: <Joint | Graph | Vector | Distribution | Kernel>
attempt_index: <reserved attempt index or null>
iteration_index: <qualified index or null>
iteration_qualification: <complete numerical receipt and allocation, or explicit not-qualified reason>
baseline_id: <baseline>
trial_id: <trial or not_applicable>
revision_id: <revision or not_applicable>
independent_of_design_and_implementation: <true | false, explain below>
status: <passed | changes_requested | not_run | unavailable>
---

# Independent review handoff

## Verdict and scope

<Verdict, covered domains and exact limits; review does not establish measured
device correctness or performance acceptance.>

## Reviewed identities and independence

| Item | Exact identity / path / check |
| --- | --- |
| Hypothesis / implementation / source freeze | <revision and hashes> |
| Actual IR / generated source / library / manifest | <hashes and pin checks> |
| Previous scope reused | <byte-identical mapping or not_applicable> |

<Actual context independence, excluded scope and uninspected evidence.>

## Findings and dispositions

| Finding ID / severity | Location / concrete failure | Contract / reproducer | Disposition / repair evidence |
| --- | --- | --- | --- |
| <ID or explicit none> | <file/symbol/span> | <causal proof> | <open/resolved, revision> |

## Independent checks

| Domain / check / actual command | Scope | Result / exit | Evidence |
| --- | --- | --- | --- |
| <check> | <CPU/source/NVCC/GPU> | <actual result> | <hash, log> |

<Affected numerical/layout/writer/effect/lifetime/full-module/invalidation/
partition/resource/integration evidence.>

| Expert check ID | Inspected checkpoint / values | Independent result / reason | Evidence |
| --- | --- | --- | --- |
| <assigned G/V/D/K check IDs> | <exact current scope or pinned unchanged lineage> | <pass/fail/not_run/not_applicable; reason> | <independent receipt; Vector type inventory when affected> |

## Expert baselines and SOTA target

| Adopted implementation / revision / license | Reused or ported symbols / mapping / memory strategy | Required numerical / ABI adaptations | Matched baseline reproduction | Improvements beyond it / measured gap |
| --- | --- | --- | --- | --- |
| <independently inspected starting point and source pin> | <verified upstream-to-local code lineage> | <reviewed adaptation diff and reason> | <inspected receipt or missing evidence> | <verified changes; source review does not establish a measured win> |

| Domain / comparison ID | Best-known baseline / source / date | Matched contract / metric / baseline result | Target / current result / gap | Evidence scope / status / next action |
| --- | --- | --- | --- | --- |
| <affected domain/ID> | <independently inspected baseline source/commit/date> | <comparability and metric check> | <verified result/proof or unsupported claim/gap> | <independent evidence; pending measured gates or specific not_applicable reason> |

## Gate coverage and limits

<Covered/pending domains, blockers, skips/not_run, unmeasured claims;
review.json/verdict record must agree.>

## Kernel microprofile

<For Kernel work: independent inspection of exact ncu report/export, raw WG and
hardware events, group/module attribution, clock/coverage/drop validity, SVG
regeneration and overlap calculations. Check measured activity versus in-flight
envelopes and instrumentation effects. State K6/K7/K10 evidence and separate
ncu/WG/hardware-interval/analysis gate results; source-only review leaves missing
runtime gates open. Follow the microprofile handoff reference; otherwise give a
specific not_applicable reason.>

## Next handoff

<Optimizer repair or Verifier evaluation; exact revision and required gates.>
