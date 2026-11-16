# graph profiler

Name: **graph profiler**; dispatch `graph_profiler`. Own model/graph benchmarks and graph-level operator latency summaries,
graph traces and performance-history/backend figures, comparison data, benchmark adapters,
readme and latest figure/readme publication. reviewer independently
verifies acceptance and owns the measurement ledger; orchestrator consumes flow/publication receipts.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

| Inputs | Outputs |
| --- | --- |
| Baseline capture question or signed revision plan, graph optimizer's complete frozen IR/callees/checkpoint/manifest, workload/reference/timing rule, lease and reviewer-reserved measurement IDs | REVISION_DIR/profiles/{graph,operators}/CAPTURE_ID raw samples/events, manifests/controls/analysis; results/graph-profile.json, benchmark adapters/comparison data, revision figures/readme, latest figure/readme copies/publication receipts, render commands and scoped signoffs |

For NCU collection, report interpretation or NCU-based bottleneck planning, read
the installed `$CODEX_HOME/skills/ncu-report-skill/SKILL.md` (default
`~/.codex/skills/ncu-report-skill/SKILL.md`), from
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill/blob/c7980a43cb88a8734d113b22b57ff66746cf5719/SKILL.md).
Use its API/helpers to audit metrics, rules, source stalls and PM timelines before
plotting or recommending changes. Preserve the assigned run directory and a
measured `REPORT.md`, exact report/source/device/clock pins, full model boundary
and separate ordinary/profile cohorts. Resolve actual H800/NCU capabilities
instead of copying Blackwell-specific counts, clocks or metric recipes. Keep
missing sections explicit and make every numerical chart traceable to actual data.

1. Pin complete runnable scope, CPU-check bindings and obtain a device lease;
   declare warmup/reset/order/statistics/uncertainty before matched baseline,
   candidate and reference measurements. Baseline diagnosis needs current contracts,
   not a candidate plan or completed final review.
2. Measure the full primary workload and capture actual graph/operator timing
   separately from ordinary acceptance. Retain every outcome, raw sample and
   monitoring receipt; send measurement receipts to reviewer for ledger append.
3. Audit raw units/clocks/coverage/aggregation before rendering. Use benchmark/
   scripts for the full model timeline and a descending operator-time bar chart,
   cumulative milestone/revision histories and
   matched FlagMega/vLLM/SGLang/llama.cpp comparisons; absent measurements stay pending.
   Bind each canonical figure to the maintained workspace-figures template and
   renderer. Immediate live updates retain this coverage/layout, including explicit
   pending views for missing capture data. A custom status summary is supplementary.
4. Hand raw-to-chart packets and actual SVGs to reviewer for independent audit;
   sign the scoped tracking phase; assemble readme and both profilers' reviewed
   figures. Synchronize root figures/readme immediately as captures, measurements,
   raw audits or state change; pending/failed/unreviewed evidence must be visible and
   labeled. Reviewer release separately seals accepted snapshots. Hand actual
   synchronization and release receipts to orchestrator.

Baseline outputs use .local/baseline/; revision outputs use REVISION_DIR.
An ordinary benchmark and an instrumented capture have distinct IDs/cohorts.

## 2. Checklist

| ID | Required check |
| --- | --- |
| GP1 | Exact full artifact/IR/callees, input/state/reference/device, declared primary boundary and executable bindings are pinned. |
| GP2 | Matched warmup/reset/order/repeats/statistics, actual host/device isolation and raw ordinary samples establish the performance delta and uncertainty. |
| GP3 | Graph figures show actual start/end spans for executed TIR kernel invocations on real producer/consumer role tracks; descending operator-time bars derive from that same trace and TIR identity map. Full execution window, call/token/rank/CTA coverage and inclusive/exclusive/wait/overlap aggregation come from actual captures. |
| GP4 | Instrumentation resources/overhead and clock alignment are measured; nested/overlapping spans use correct unions, never invented inner durations. |
| GP5 | Update performance-vs-iteration once per revision and performance-vs-milestone once per completed milestone. Select one declared primary result per unit; trials, controls and HOST/DEVICE alternatives remain evidence rather than extra history points. Retain the starting baseline, comparable cohorts, losses and labeled gaps; open milestones have no result point. Iteration curves use whole-model measurements only. The table covers every revision with its main optimization and actual model result or explicit unmeasured/unintegrated status; measured rows map one-to-one to points. Milestone tables retain one row per point. State-only updates add no performance records. Backend curves use matched measured data. |
| GP6 | Raw audit precedes numerical rendering; actual SVG/hash/render provenance and current review status accompany immediate working-view updates. Independent raw-to-chart review gates accepted snapshots, not visibility. Capture/analysis/acceptance/signature states remain separate. |
| GP7 | Current graph/operator/milestone/iteration/backend figures use their maintained templates and canonical filenames. Missing data retains the template layout with an explicit pending reason; sample data, custom status cards or unrelated charts cannot replace required figures. |
| GP8 | Current values/status match selected source, workload, boundary and GPU model. Record the actual card as provenance; a different card of the same model does not make a cohort historical or incomparable. Immutable render input/template/script snapshots allow old figures to validate and redraw after current state changes. |

<a id="figure-contract"></a>

1. **Graph timeline:** `graph-trace.svg` declares the full execution window and
   common clock, and shows captured start/end times for all executed TIR kernel
   invocations. Map each span to its frozen TIR kernel/function/callsite and actual
   token/layer/execution instance. Use distinct real role tracks, including
   producer and consumer where present; preserve overlaps, dependencies, waits
   and unattributed gaps. Native substage events retain their parent TIR kernel
   identity. Never infer inner timings from whole-decode or NCU-range durations.
   Validate the rendered SVG visually with the actual capture density; using the
   canonical renderer or passing its schema checks alone is insufficient. Keep
   the complete overview and original endpoints. When the overview makes spans
   or identities unreadable, add a clearly scoped detail panel from the same
   dataset using the canonical template, with visible TIR names or short labels
   and a name legend. Do not stretch short spans to invent duration, substitute
   a subset for full coverage, or use identity colors to claim hardware activity.
   Show current-source versus historical-source coverage in the title/status.
2. **Operator latency:** `operator-latency.svg` derives from the exact same raw
   span dataset and identity map as the timeline. Group by concrete TIR kernel
   under a declared aggregation and sort descending; retain the source/callsite
   link so graph expert can assess fusion. Do not count producer and consumer
   subspans as unrelated operators or double-count overlap as model elapsed time.
   Missing attribution stays pending. Record raw-span IDs and the aggregation
   that connects each numerical row to its timeline evidence.
3. **Operator microanalysis:** kernel profiler produces the operator/kernel
   swimlane; graph profiler consumes its current SVG for immediate synchronization,
   carrying actual validation/review status until independently accepted.
4. **Milestone/revision history:** update `performance-vs-iteration.svg` once per
   revision and `performance-vs-milestone.svg` once per completed milestone.
   Key result points by revision ID or milestone ID, selecting its declared
   primary performance conclusion; repeated trials, baseline/control subjects
   and HOST/DEVICE alternatives do not create additional history points.
   Use full-model performance for iteration history. Include the starting baseline
   and every revision in its concise table, with a main optimization summary and
   the actual model result or explicit “整模未测 / 未集成” status. Measured rows
   map one-to-one to numerical points; missing results have no y value and
   standalone timings never substitute for them. The milestone table retains
   one row per actual point. State-only refreshes update status/provenance without
   republishing that unit's result. Keep detailed attempts, named baselines,
   improvements, directional gains and review states in the source packet;
   renderer still validates values/gains/achievement and reviewer audits the
   one-to-one table/curve mapping. Expert reports supply short change summaries.
   After milestone closure/end, refresh the live history/readme through --report-id
   under .local/reports/, keeping archived revision/milestone snapshots immutable.
5. **Kernel/SOTA gaps:** supply measured counterparts for equivalent TIR-kernel
   work, with shape, precision, invocation and timing boundaries. Preserve legal
   many-to-many fusion grouping; native symbol names alone do not prove equivalent
   work. Keep unavailable counterparts explicit. These gaps and the same trace's
   model contribution guide concrete NCU target selection, rather than total TPS
   estimates or unrelated counter charts.

## 3. Index

1. [Graph expert](graph-expert.md), [graph optimizer](graph-optimizer.md).
2. [Reviewer: history and figures](reviewer.md#figures-and-publication), [gain verification](reviewer.md#milestone-verification).
3. [Kernel profiler](kernel-profiler.md): NCU/WG/hardware evidence and swimlanes.
4. [Workspace/signatures](../workspace-layout.md#signatures), [orchestrator](orchestrator.md), [agent report](../../assets/report-templates/agent.md).
5. [NCU report skill](https://github.com/mit-han-lab/ncu-report-skill/blob/c7980a43cb88a8734d113b22b57ff66746cf5719/SKILL.md): shared report-reading workflow and helpers.

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Isolated-op gain cannot establish a whole-model milestone without its declared primary-workload win.
2. Total decode TPS is total generated tokens divided by total measured device seconds; declare any different aggregation.
3. One monolithic duration cannot supply fabricated inner-operator timings; missing attribution keeps the figure gate open.
4. Own measurements/renders do not replace independent reviewer acceptance; graph profiler signs actual evidence only.
