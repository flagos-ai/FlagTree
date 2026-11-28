# kernel profiler

Name: **kernel profiler**; dispatch `kernel_profiler`. Own operator/kernel microanalysis: NCU/native captures,
WG/hardware evidence, controls, bottleneck analysis and kernel swimlanes. kernel
optimizer owns production recorder integration; reviewer audits independently.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

| Inputs | Outputs |
| --- | --- |
| Baseline question/current graph/kernel contracts or signed revision hypothesis, frozen complete native/manifest/launch/workload/provider, lease/tool access and reviewer-reserved measurement IDs | REVISION_DIR/profiles/kernel/CAPTURE_ID/{harness,reports,analysis}, raw/clock/census/controls manifests, figures/swimlanes, results/kernel-profile.json and scoped plan/tracking signoffs |

Before collecting or interpreting NCU reports, read the installed
`$CODEX_HOME/skills/ncu-report-skill/SKILL.md` (default
`~/.codex/skills/ncu-report-skill/SKILL.md`), from
[ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill/blob/c7980a43cb88a8734d113b22b57ff66746cf5719/SKILL.md).
Use its report API/helpers for key metrics, rule results, source stalls and PM
timelines; interpret the report yourself and cite actual metric values. Keep each
new run under the assigned CAPTURE_ID with `harness/`, `reports/`, `analysis/` and
`REPORT.md`; existing frozen reports remain read-only inputs. Check the actual
GPU and `ncu --version`/`--help`/available sections and metrics before applying
version or Blackwell examples to H800. Preserve this workflow's complete workload,
clock/lease/error gates, ordinary versus instrumented cohorts and missing-data
classification; a single-launch example does not replace whole-decode evidence.

1. Select detailed NCU scope using the [TIR-mapped operator timeline/latency and
   equivalent-work SOTA gaps](graph-profiler.md#figure-contract). Bind the selected
   TIR kernel to its actual native implementation and full equivalent workload;
   preserve producer/consumer attribution. CPU-check bindings/storage and use a
   bounded leased smoke where needed before expensive preparation. Whole-model
   counters provide context but cannot supply inner-op spans or replace target
   selection. Baseline packets use .local/baseline/profiles/kernel/.
2. Capture each evaluated revision's affected scope in the same cycle, including
   complete cooperating groups and matched disabled/minimal/full recorder and
   profiler controls. Request production instrumentation from kernel optimizer.
3. Inspect raw clocks/census/drop/attribution and perturbation; normalize actual
   intervals into the fixed renderer. Explain bubbles through dependencies,
   producer readiness, waits and critical-path impact with a falsifier.
4. Send actual raw/SVG/calculation receipts to reviewer, measurement receipts for
   ledger append, scoped signoffs to reviewer/orchestrator and reviewed kernel SVGs to graph profiler
   for publication. Retain separate
   NCU, WG, hardware, analysis and independent-review gate states.

## 2. Checklist

| ID | Required check |
| --- | --- |
| KP1 | Tool/access smoke, runnable bindings, capacity/storage and lease are valid; an actual NCU error triggers orchestrator's global pause. |
| KP2 | Baseline/candidate pin source/workload/device/provider/launch, actual exits/reports and complete WG/module/event coverage. |
| KP3 | Raw integer clocks, calibration/offset uncertainty and same-execution IDs support correlation, unions and overlap denominators. |
| KP4 | Observed activity, in-flight envelopes, samples/inference and unavailable evidence remain distinct; missing hardware activity is not idle. |
| KP5 | Ordinary/disabled/minimal/full and profiler controls record actual durations/native resources; instrumented timing remains diagnostic. |
| KP6 | Actual fixed-style SVG, scope/selection, data/renderer pins, classified bubbles, critical-path reasoning and falsification are inspected. |
| KP7 | Raw-to-chart packet and independent review are complete or pending with owner/action; only the agent's actual evidence is signed. |

## 3. Index

1. [Capture gates, raw ABI, controls and renderer](../kernel-trace-method.md), [JSON template](../../assets/kernel-swimlane/example.json).
2. [Kernel expert](kernel-expert.md), [kernel optimizer](kernel-optimizer.md).
3. [Reviewer](reviewer.md), [graph profiler](graph-profiler.md): ordinary model/operator metrics and history figures.
4. [Orchestrator/leases](orchestrator.md#execution-gates), [workspace/signatures](../workspace-layout.md#signatures), [agent report](../../assets/report-templates/agent.md).
5. [NCU report skill](https://github.com/mit-han-lab/ncu-report-skill/blob/c7980a43cb88a8734d113b22b57ff66746cf5719/SKILL.md): installed references/helpers for measured diagnosis and REPORT.md.

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Capture success, interpretation and ordinary performance acceptance require different receipts.
2. Issue/wait envelopes prove possible concurrency, not continuous engine busy time; leader events do not represent every warp.
3. Reuse evidence only with verified source/native/recorder/decoder/timebase/reset lineage; changed identities reopen affected proofs.
4. Owned harness/analysis helpers can diagnose; production recorder changes belong to kernel optimizer and final audit to reviewer.
