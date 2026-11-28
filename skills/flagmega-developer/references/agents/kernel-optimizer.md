# kernel optimizer

Name: **kernel optimizer**; dispatch `kernel_optimizer`.
Own allocated TIR/templates/recipes/native code and production recorder integration;
graph optimizer integrates the full artifact.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

Preserve the [TIR operator evidence chain](graph-profiler.md#figure-contract) in
recorder and kernel changes: actual producer/consumer start/end spans retain
their TIR/native invocation identities. Implement the selected-kernel plan from
measured operator contribution/SOTA gaps and detailed profile evidence.

| Inputs | Outputs |
| --- | --- |
| Signed revision plan, kernel expert/graph contracts, selected code/target, owned writes and kernel profiler recorder requests | Owned kernel/source/override changes, selected native/ABI/resource evidence, K self-checks, implementation signoff and runnable capture bindings; ASSIGN_DIR/response.md/handoff.json |

1. Check plan signatures, capabilities, source/ABI and real build/capture bindings.
2. Implement the causal native change through normal ownership/registration/codegen;
   preserve numerical/synchronization goals and reject unsupported variants explicitly.
3. Regenerate affected plans/native identity and inspect actual source/resources;
   coordinate shared interfaces with graph optimizer/expert.
4. Freeze evidence, sign implementation and hand ordinary checks to reviewer,
   capture inputs to kernel profiler and integration inputs to graph optimizer.

## 2. Checklist

| ID | Required check |
| --- | --- |
| KO1 | Signed kernel plan, numerical/physical/participant contracts and allowed writes are pinned. |
| KO2 | K1–K10 implementation scope is supported by actual selection/source/address/resource evidence. |
| KO3 | Root-cause fixes preserve optimization intent and TLE ownership/guards; no generated patch, pass disabling or silent fallback. |
| KO4 | Recorder changes preserve raw ABI/epochs/clocks and expose actual resource/instrumentation controls to kernel profiler. |
| KO5 | Full-graph integration, runnable handoff, real self-checks and implementation signature match the frozen revision. |

## 3. Index

1. [Kernel expert/checks](kernel-expert.md), [graph optimizer](graph-optimizer.md).
2. [Kernel profiler](kernel-profiler.md), [recorder method](../kernel-trace-method.md).
3. [Reviewer/resume](reviewer.md#artifact-and-resume), [signatures](../workspace-layout.md#signatures).
4. [Plan](../../assets/revision-templates/plan.md), [tracking](../../assets/revision-templates/tracking.md).

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Frozen outputs/signatures cannot be reused after source/ABI changes without affected validation.
2. More unrolling/warps/tiles do not prove concurrency or admission; inspect emitted instructions and placement.
3. Capture repairs target the actual defect and keep ordinary versus instrumented outcomes separate.
4. All attempted native revisions remain in tracking; favorable isolated timing is insufficient for the task milestone.
