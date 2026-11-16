# Revision tracking

## 1. Frozen identity

| Field | Value / pinned receipt |
| --- | --- |
| run_id / milestone_id / revision_id | <run / mNNN / rNNN> |
| plan_version / plan_hash / plan approvals | <frozen plan body and authentic receipt pins> |
| baseline / source / IR / generated / library / manifest | <exact identities and hashes> |
| attempt_index / iteration_index / NUM qualification | <reserved attempt; iteration null until complete NUM/allocation receipt> |
| Required phase signatures / gates | <applicable agents and signed plan scope> |

## 2. Append-only events

| Sequence / UTC | Actual agent / assignment | Action / command / input scope | Outcome / evidence / next owner-action |
| --- | --- | --- | --- |
| <seq / UTC> | <canonical agent and actual context> | <exact action and pinned inputs> | <pass/fail/not_run/unavailable; durable receipt> |

<reviewer appends events, maintains phase snapshots and transcribes actual agent receipts. Retain failed attempts,
plan changes, invalidated approvals and corrections; never rewrite an old verdict.>

## 3. Measurements and revision outcome

| Scenario / measurement IDs | Baseline / candidate / unit / direction | Gain / uncertainty / validity | Required gate / evidence / pending owner |
| --- | --- | --- | --- |
| <matched constituent ledger IDs> | <actual values; unavailable needs reason> | <measured delta and predeclared rule verdict> | <review/NUM/resources/capture/analysis/figures/reproduction> |

Revision outcome: <improved / no_gain / rejected / invalid / pending>.
Milestone status: <open until clear primary-workload gain and all applicable gates/signatures>.
Selected winning revision: <unset or actual achieved_revision in milestone.json>.

## 4. Agent signatures

Frozen phase subject: <sequence and results/tracking-snapshots/SEQUENCE.md SHA256,
excluding this section; plan/source/artifact-manifest and evidence pins>.

| Agent / actual context / assignment | Phase / decision | UTC | Signed subject / evidence pins | Own handoff receipt / SHA256 |
| --- | --- | --- | --- | --- |
| <applicable optimizer/expert/profiler/reviewer/orchestrator> | <implementation / assessment / measurement / analysis / verdict / reconciliation; pending until signed> | <UTC> | <revision/plan/source/manifest/phase snapshot pins> | <agent-authored receipt/hash> |

<Only the actual agent signs its scope. Later events do not renew older phase
approval; changed plan/source/artifacts need renewed affected signatures. Numeric
pass, build completion and noise-level gain cannot achieve a milestone. Follow
[signature rules](../../references/workspace-layout.md#signatures) and the
[gain gate](../../references/agents/orchestrator.md#milestone-gate).>
