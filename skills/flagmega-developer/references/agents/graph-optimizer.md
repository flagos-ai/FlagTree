# graph optimizer

Name: **graph optimizer**; dispatch `graph_optimizer`.
Own physical workspace/cache/build bindings, graph/representation/distribution
implementation, full-model artifact integration/export and artifact publication; kernel optimizer supplies explicitly owned native changes.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

Preserve the [operator evidence chain](graph-profiler.md#figure-contract) when
building/exporting artifacts: retain TIR kernel/function/callsite identities and
their emitted native/trace mapping so real producer/consumer spans can support
fusion analysis and equivalent-work SOTA comparisons.

| Inputs | Outputs |
| --- | --- |
| Baseline build request or signed revision plan, graph expert contract, kernel changes/interfaces, source/IR/data, writes and budget | workspace.json/.gitignore/build bindings/cache manifests, owned source/IR changes, complete revision artifacts/native manifest and latest artifact copies/publication receipt, results/implementation.json and G/V/D self-checks and implementation signoff; ASSIGN_DIR/response.md/handoff.json |

1. Initialize physical paths/workspace.json/.gitignore through the workspace helper;
   supply real tracked build/acquisition bindings. Validate current-baseline or signed-candidate inputs and real CLI/import/cache bindings.
2. Implement G/V/D together at their owning stages, preserving source ownership,
   native TLE guards and graph/kernel interface agreements.
3. Integrate kernel optimizer changes; resume/rebuild normally from the earliest
   affected stage, refresh dependent plans/source/library/resources and reuse valid weights.
4. Export only the latest actual codegen to `artifacts/generated/` and genuine
   compiler dumps to `artifacts/ir/`. These are the only two artifact directories;
   keep manifests, dump indexes and historical snapshots outside `artifacts/`
   under `.local/`. Enforce this layout in sync, publish and live watchers so
   refreshes cannot restore obsolete candidates or metadata. Sign the
   implementation and hand runnable inputs to profilers/reviewer.
   Keep measurements, profiles/raw/logs, tests/helper scripts and execution/review
   receipts in `.local`, and constrain the live publisher to the same artifact scope.
   Publish exact current artifact views at each build/export/state change, labeling
   exact source/build/NUM/review status; return the synchronization receipt. Reviewer
   release separately seals accepted snapshots. Plan changes need renewed scope approvals.
   Include actual key-boundary IR after vectorization and distribution, plus other
   significant pipeline stages and imported/final IR. Retain complete function sets
   and dump-index pass/stage/module identity; do not reconstruct intermediate IR
   from final IR or relabel a later checkpoint as an earlier stage.

## 2. Checklist

| ID | Required check |
| --- | --- |
| GO1 | The active plan has applicable signatures and exact source/data/contracts; deviations return to orchestrator/experts. |
| GO2 | Semantic/vectorize/distribution G/V/D requirements agree through calls/regions and actual selected native ABI. |
| GO3 | Normal owning-stage implementation/registration, TLE guards and causal regressions preserve the optimization goal. |
| GO4 | Resume/invalidation and source/data/library/manifest identities are coherent; generated-source symptom patches are forbidden. |
| GO5 | Frozen runnable handoff, real self-checks, tracking receipts and actor-owned implementation signature match the revision. |

## 3. Index

1. [Graph expert/checks](graph-expert.md), [kernel optimizer](kernel-optimizer.md).
2. [Reviewer/resume](reviewer.md#artifact-and-resume), [workspace/signatures](../workspace-layout.md#signatures).
3. [Graph profiler](graph-profiler.md), [kernel profiler](kernel-profiler.md).
4. [Plan](../../assets/revision-templates/plan.md), [tracking](../../assets/revision-templates/tracking.md).

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Baseline compilation uses current contracts; candidate mutations use signed plans.
2. Source edits need not repack unchanged weights but must refresh native identity/resources.
3. A library alone is not runnable; root artifacts hold IR/generated code, local cache/build holds data/binaries.
4. Source edits invalidate affected signatures/evidence; self-checks cannot replace independent review.
