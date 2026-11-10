# <Model × workload × hardware> optimization trace

## Contract and current status

| Field | Value |
| --- | --- |
| Run / task | <run ID, requested objective> |
| Input / numerical contract | <checkpoint identity, dtype/rounding, state, shapes> |
| Hardware / backend / reference | <device, runtime, clean reference revision/config> |
| Primary metric / acceptance | <units, timing boundary, scenarios and rule> |
| Iteration qualification / counters | <complete numerical pass is required; attempt IDs remain separate; qualified baseline 0, candidate indices and receipts or none> |
| Current baseline / candidate / accepted best | <exact IDs, accepted artifact or none> |
| Current status / unmet gates | <actual status, independent review/device/performance/reproduction> |
| Public data / iteration figure | <Git-managed links or not_run reason> |

## Iteration index

| Entry ID | Attempt | Iteration | Kind | Baseline / trial / revision | Status | Primary result |
| --- | --- | --- | --- | --- | --- | --- |
| <stable ID, link to entry> | <attempt index or not_applicable with reason> | <qualified index or not_assigned with reason> | <kind> | <IDs> | <status> | <metric or not_run reason> |

## Entries

<Insert every meaningful entry using trace-entry.md in actual order. Preserve
baseline failures, repairs, regressions and rejected/unaccepted candidates.>

## Remaining work and final handoff

<Next action, all unmet criteria, accepted exact artifact/reproduction only when
proven; iteration dataset/figure coverage and any explicitly omitted scope.>
