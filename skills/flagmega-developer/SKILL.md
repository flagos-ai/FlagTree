---
name: flagmega-developer
description: Optimize FlagMega models through eight scoped agents, signed revisions and measured performance milestones.
---

# FlagMega developer

Use this workflow to establish correctness and performance baselines,
profile bottlenecks, and validate FlagMega model optimizations. The
[FlagMega tutorials](../../python/tutorials/flagmega/README.md) cover
NVIDIA and AMD. Nsight Compute (NCU) guidance applies to NVIDIA GPUs.

## 1. Workflow

Orchestrator maintains `WORK_DIR/dashboard.md` as the sole management record,
using a table of when work happened, completed work, current work, remaining work,
owners, flow steps, evidence and prerequisites. Update the table from authoritative
receipts before dispatching or advancing. Manage only from this table and the
numbered flow; do not create or use a separate state.json or JSON management mirror.

1. Freeze objective, reference, workload, environment and acceptance; establish a
   measured baseline. Synchronize root artifacts/figures immediately as real outputs
   or state change, including pending, running and failed work. If the baseline
   already meets the goal and all acceptance, finalize its reviewed snapshot and end.
2. Open a milestone/revision with draft plan.md and tracking.md; profile current
   inputs, let experts analyze and finalize the plan, then collect applicable signatures.
3. Implement and evaluate the revision, retaining all attempts, actual checks,
   profiler evidence and independent review signatures.
4. Freeze each revision's declared primary result and its actual acceptance state.
   A new independent optimization hypothesis starts a new revision, even when the
   previous revision still has pending final acceptance. Repairs, repeated trials
   and missing checks for the same hypothesis stay with their owning revision.
   A clear measured win with complete acceptance closes the milestone; reaching
   the full user goal and acceptance ends the task. See the
   [revision boundary and transition rules](references/agents/orchestrator.md#revision-boundary).

Root artifacts/figures always show the latest working state. Build/export updates
artifacts; captures, measurements and state transitions update figures in the same
cycle. Mark source identity and actual build/NUM/measurement/review status visibly;
missing data stays pending, never zero or synthetic. Reviewer release seals accepted
snapshots and achievement; it does not delay current-state visibility. Preserve
immutable historical snapshots when the working view changes.
Artifacts include real IR at key compiler boundaries, especially after vectorization
and distribution, with pass/stage identity and hashes, as well as imported/final IR.
`artifacts/` has exactly two directories: `generated/` for the latest actual
codegen and `ir/` for the latest genuine IR dumps. No other files or directories
belong there, including manifests or historical candidates. Keep source/stage
manifests, dump indexes, historical snapshots, measurements, profiler reports,
raw data, logs, tests, helper scripts and receipts outside it under `.local/`.
Publishers replace the current payload instead of accumulating revision history.
Latest figures follow the maintained figure templates and canonical coverage in
the workspace map. A status summary supplements those figures; it does not replace
them. Missing captures use the corresponding template's explicit pending layout,
with no invented timelines, numerical points or synthetic example data.
`performance-vs-iteration.svg` receives one result update per revision;
`performance-vs-milestone.svg` receives one result update per completed milestone.
Use one declared primary performance result for that unit, plus the starting
baseline. Trials, repeated measurements, controls and HOST/DEVICE alternatives
remain in the evidence packet, not separate history points. Open milestones and
unmeasured revisions have no numerical result point. The iteration table includes
the baseline and every revision, with its full-model result or explicit
“Full model unmeasured / not integrated” status and main optimization. Each measured row maps to one
full-model point; operator timings cannot substitute for model results. The
milestone table has one concise row per plotted point. State changes refresh
status and provenance without creating another performance record for the unit.

## Operator evidence and optimization decisions

Figures are inputs to optimization decisions. `graph-trace.svg` shows captured
start/end times for every executed TIR kernel invocation, mapped back to its TIR
source/callsite, on real role tracks such as producer and consumer.
`operator-latency.svg` derives its descending ranking from the same trace and
identity map; a whole decode interval cannot supply inner-operator durations.
Graph expert uses these spans and dependencies to assess useful legal fusion.
Compare equivalent TIR-kernel work with measured SOTA counterparts, preserving
shape, precision, boundary and fusion differences. Orchestrator uses the actual
kernel gaps and model contribution to select a concrete kernel for detailed NCU
profiling and optimization. Extra whole-model counter charts do not replace this
data chain. Missing timings or counterparts stay explicit, without invented values.
See the [operator figure contract](references/agents/graph-profiler.md#figure-contract).

## GPU model equivalence

Measurements on different cards of the same GPU model are equivalent for this
workflow. Use any idle card of the requested model; GPU index, UUID and PCI
address do not define a separate comparison cohort or require a fixed card.
Reuse valid NUM/state/resource/profile evidence across that model when the
workload, precision, implementation, toolchain and measurement boundary match.
Do not rebuild, repeat qualification or request new technical plan/source reviews
solely because execution moves to another card of the same model.

Record the actual card in each execution receipt and bind its routing, idle check
and lease before launching. Runtime routing metadata may be refreshed without
invalidating the reused technical evidence. Raw clock correlations and timeline
overlaps still require the original execution's clocks and calibration.

## 2. Checklist

1. Use only the eight registered types. orchestrator controls flow; delivery owners
   produce technical details. Name inputs, writes, templates/scripts and recipients.
2. Every revision has a versioned plan, append-only tracking and authentic scoped signatures.
3. A numerical pass, completed build, theoretical bound or noise-level delta cannot
   establish milestone achievement. Preserve failed and no-gain revisions.
4. Keep baseline/candidate comparisons matched and all required gates visible;
   close only after independent verification and reproduction.
5. Synchronize current artifacts/figures on every authoritative output/state change,
   including baseline establishment and errors; verify root freshness before handoff.
6. Dashboard rows and flow determine assignments, leases, gates and transitions.
   Requests name their dashboard row IDs and flow steps; all agents read the table,
   only orchestrator writes it.

## 3. Index

1. [Numbered optimization cycle and eight-agent registry](references/agents/orchestrator.md),
   [workspace, revisions and signatures](references/workspace-layout.md).
2. [Graph expert](references/agents/graph-expert.md), [graph optimizer](references/agents/graph-optimizer.md),
   [kernel expert](references/agents/kernel-expert.md), [kernel optimizer](references/agents/kernel-optimizer.md).
3. [Graph profiler](references/agents/graph-profiler.md), [kernel profiler](references/agents/kernel-profiler.md),
   [reviewer and measurement history](references/agents/reviewer.md).
4. [Delivery templates/scripts](assets/workspace-templates/deliverables.json), [workspace helper](scripts/workspace.py),
   [revision plan](assets/revision-templates/plan.md), [revision tracking](assets/revision-templates/tracking.md),
   [agent report](assets/report-templates/agent.md), [trace document](assets/report-templates/optimization-trace.md),
   [trace entry](assets/report-templates/trace-entry.md), [trace validator](scripts/validate_trace.py).
5. [NCU report skill](https://github.com/mit-han-lab/ncu-report-skill/blob/c7980a43cb88a8734d113b22b57ff66746cf5719/SKILL.md),
   installed under `$CODEX_HOME/skills/ncu-report-skill` (default `~/.codex/skills`).
   Both profiler roles read it for NCU collection/interpretation while retaining
   FlagMega's full workload, resource gates, provenance and performance-point rules.

## 4. Lessons

1. A planned milestone and an achieved milestone are different states.
2. Reuse evidence by identity; changed plans/source/artifacts need affected signatures again.
3. Start ready work under its own prerequisites; separate measurement from rendering/review.
4. Keep performance gaps, failures and unsigned work assigned instead of declaring completion.
