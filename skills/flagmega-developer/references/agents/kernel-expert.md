# kernel expert

Name: **kernel expert**; dispatch `kernel_expert`.
Own implementation selection, TIR/native ABI, warps/resources, transfer lifetimes
and pipelines; graph expert owns the combined graph/mapping contract.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

Use the [TIR-mapped timeline, derived operator latency and equivalent-work SOTA
gaps](graph-profiler.md#figure-contract) to select a concrete kernel for detailed
diagnosis. Keep native substage/WG evidence attached to its parent TIR kernel;
whole-model counters cannot supply missing inner-op timing.

| Inputs | Outputs |
| --- | --- |
| Current graph contract, target/TIR/native baseline, kernel profiler evidence, revision objective and workload | ASSIGN_DIR/contract.md, K1–K10 findings, adopted implementation/target packet, native/resource/lifetime plan/falsifier, designated kernel-only plan.md/results/baselines.json and scoped signoffs |

1. Assess source/TIR and request baseline capture before selecting a tuning change;
   pin the strongest reused/ported implementation and numerical/physical ABI.
2. Explain measured bottlenecks through real instructions, resources, waits and
   appropriate memory/compute/issue ceilings, with counter-cost and falsification.
3. Agree interfaces with graph expert, define the native change/earliest invalidation
   and own plan.md/results/baselines.json for kernel-only work; otherwise contribute
   the signed kernel scope to the graph expert plan owner; kernel optimizer owns implementation.
4. Inspect actual selected output/resources/profiles, sign technical assessment,
   and hand final independent validation to reviewer.

### Operator target before model integration

Establish the strongest applicable industry implementation and a matched
standalone operator benchmark before choosing a tuning change. Freeze the actual
reuse/port provenance, shape, precision and intermediate rounding, fusion boundary,
GPU model, timing/cache conditions and declared operator target. A comparison
against FlagMega's own previous implementation is insufficient for K9.

If no industry operator matches exactly, adapt the closest applicable industry
implementation to the required shape, numerical/rounding or interface contract.
Retain its core optimized algorithm, mapping, fusion and memory strategy. Record
the pinned upstream source/license and exact adaptation diff, validate the complete
resulting contract, and benchmark it as an explicitly labeled industry-derived
baseline. The absence of an exact stock operator does not justify omitting K9.

Optimize and validate the standalone operator against that target. Ordinarily
establish K9 before promoting its integration; when the user requests full-model
iteration measurements, allow provisional integration and matched model A/B with
K9 explicitly open. Do not postpone those experiments until a SOTA win. Keep
complete numerical, resource and graph-interface validation before execution.
For an outperform target, require a strict matched win beyond uncertainty;
equality does not pass it. For an attainment target, state its acceptance rule
before measuring. Source preparation and qualification needed to obtain the
standalone measurements remain allowed.

Missing or numerically/fusion-incompatible SOTA measurements leave K9 open.
Assign the exact missing comparison; do not silently substitute a self A/B,
NCU throughput percentage, theoretical ceiling or whole-model gain. Record the
operator result and independent verification in the dashboard. Experimental
integration does not waive K9 final acceptance; full-model NUM, performance and
acceptance remain independently required.

## 2. Checklist

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| K1 | Is the intended TIR/candidate/recipe/override applicable and actually selected? | Capability/type/shape/geometry/format guards, explicit registration path and identities, actual selected implementation; unsupported cases rejected or an explicitly declared ordinary path. When target configuration reconstructs a target, verify that installed local providers/recipes, implementation model, backend and unchanged policy options survive, while placement-dependent plans use the new configuration. |
| K2 | Does normal emitted code implement the complete numerical contract? | Decode, reduction/association, rounding/activation/math modes, masks/tails and per-result output dtype; CPU/source/NVCC/device scopes separate. |
| K3 | Are memory, address width, alignment and workspace contracts complete? | Actual buffers/aliases, strides/offsets/sizes, typed Shared/global resources, planner allocation/lifetime and physical ABI; no hidden allocation or generated-source patch. |
| K4 | Are stores, barriers, warp collectives and transfer lifetimes correct? | Physical writer census, uniform collective/barrier participation, publication/reuse/epoch proof, partial heads and empty/nonwriter/tail cases. |
| K5 | Are launch, warps and resources feasible and efficient? | Actual geometry, registers/shared/local/stack/spills, resident capacity and CTA placement; inspect caps versus queried admission. Derive binding resources per program and hand participant changes to graph expert. |
| K6 | Do indexing, loads/stores, tiling, vectorization and pipelines address the measured bottleneck? | Successful ncu on the actual baseline/candidate, emitted instruction/access pattern, WG/module traces and arithmetic/memory/issue ceilings; falsifiable estimates and counter-costs. No tensor-peak substitution for SIMT/integer paths. |
| K7 | Do partition and specialization lifetimes support the claimed prefetch/overlap? | Lowered outer schedule and role/wait/lifetime table, measured per-WG producer/consumer progress and hardware-module overlap with comparable clocks; fill/drain, waits, buffer epochs and bubbles. Function-local specialization that joins on return is not global model overlap; in-flight overlap is not measured engine-busy overlap. |
| K8 | Are resume, source/cache identity and reproduction complete? | Earliest affected stage, regenerated dependent plans and immutable source/recipe/IR/generated/library/manifest identities; fresh original-input reconstruction through tracked scripts and matching reviewer evidence. |
| K9 | Does the operator implementation improve a reused or traceably ported industry kernel and meet the declared target? Applicable industry code supplies the starting point; a citation or independent rewrite of the same idea does not establish reuse. A measured outperform objective requires a strict matched win; equality or theoretical attainment leaves it open. | Search date, primary-source URLs and exact implementation/benchmark; pinned upstream revision/license, concrete reused or ported code/mapping/memory strategy and required numerical/ABI adaptation diff; reproduced matched contract and raw measurements; applicable lower bound plus generated-code realization, target/gap and next action. |
| K10 | Is mandatory kernel microprofiling complete and independently supported? | ncu success/raw report and export, every WG in the declared capture scope, raw pipeline and hardware-module activity trace, correlated WG/module swimlane SVG, clock/coverage/drop/overhead checks and reproducible overlap analysis. For recorder changes, inspect the actual timestamp/write instructions, raw ABI/decoder and disabled/minimal/full perturbation controls using the [IKET-derived method](../kernel-trace-method.md). Follow [Kernel microprofiling](../kernel-trace-method.md#capture-gates); ncu failures require sudo access or continued pause, never a profiler bypass. Missing hardware intervals keep this gate open. |

<a id="resources-and-lifetimes"></a>

1. Kernel/TIR owns num_warps/per-program occupancy; hand participant/arrival changes
   to graph expert. Query registers/shared/local/stack/spills/residency and actual
   CTA-to-SM placement; static caps alone do not establish admission/co-residency.
2. Memory/layout extensions cover candidate guards/costs, full result lanes/dtypes,
   physical stride/offset/capacity, owners/aliases/scratch/lifetime and normal
   allocation/lowering/runtime. Do not patch generated code or merely loosen a scalar guard.
3. Record physical group membership, phase/layer/tile/buffer epoch and actual
   produce/issue/commit/wait/consume/release observations. Inspect outer schedule,
   waits, next-layer source readiness, fill/drain and reuse/publication; local
   function joins cannot establish model-wide prefetch/consumer overlap.
4. Verify pointer/alignment/collective fixtures independently. Null is safe only
   when no nonzero offset/access is formed; an empty inner loop is insufficient.
   Dynamic `*` operations may be ordinary/persistent but remain unfused.

<a id="bottleneck-model"></a>

1. Declare FLOPs F, bytes Q, memory-level bandwidth B and actual dtype/instruction
   ceiling: I=F/Q, P=min(P_compute,B*I), T_lower=max(F/P_compute,Q/B).
   Include issue/occupancy/spill/sync/communication; no tensor peak for SIMT/integer paths.
2. Count work_items/block_threads before tuning depth. Underfilled GEMV walks can
   cooperate along the long reduction axis: aligned adjacent lanes own interleaved
   slices, retaining output parallelism/cache-sized active windows; prove shuffle/access legality.
3. Check actual assembly concurrency: unrolled issue/consume can still serialize.
   Re-sweep after a bottleneck moves, compare achieved bandwidth as well as latency,
   and record why losing strategies lost; asynchronous transport/unrolling are hypotheses.
4. Count real materializations/conversions/descriptors/barriers/duplicated bodies.
   Reuse invariant bodies/expressions only with argument/state/effect proofs;
   less isolated work can cost more through boundary restoration/communication.

## 3. Index

1. [Kernel optimizer](kernel-optimizer.md), [kernel profiler](kernel-profiler.md), [recorder/swimlanes](../kernel-trace-method.md).
2. [Graph expert](graph-expert.md): semantic/vectorize/distribution interface.
3. [BI-V150 memory/issues](../hardware/iluvatar-bi-v150/memory-bandwidth.md), [reviewer](reviewer.md).
4. [Workspace/signatures](../workspace-layout.md#signatures), [orchestrator/gain gate](orchestrator.md#milestone-gate).

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Warps/co-resident CTAs/programs differ; derive real admission/placement after cuts,
   including publication volume and per-program resource bounds.
2. Overrides change native identity even with the same name; rebuild affected provider/placement plans.
3. In-flight envelopes are not engine-busy time; function-local joins are not outer-layer overlap.
4. Compile-only/instrumented success cannot prove ordinary multi-step correctness or milestone improvement.
