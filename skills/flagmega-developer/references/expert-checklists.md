# Expert checklists

Use these checks for domain analysis, implementation self-checks and independent
review. Architect diagnoses and proposes; Optimizer implements and supplies
evidence; Reviewer independently checks the current artifact. A source-domain
pass does not establish Verifier's device or performance acceptance.

At baseline, the main Architect obtains and consolidates four separate domain
assessments using the [expert dispatch and workspace contract](expert-workspaces.md).
Later assignments cover the affected contracts, their producers/consumers and reachable callees.
Name the inspected checkpoints and values; use the check IDs in the role report.
Every check in an assigned domain gets a result and evidence, or an explicit
`not_applicable` reason. Unchanged scope may reuse exact pinned evidence only
after confirming the relevant types, source, IR, ABI and plans remain unchanged.
Missing evidence is `not_run` or `unavailable`, never an implicit pass.

Each assigned performance domain also follows
[Expert baselines and SOTA targets](expert-baselines.md) every revision. Its
baseline/target check requires source research, a matched comparison and
attainment or gap evidence; a diagnostic or correctness-only assignment gives
an explicit `not_applicable` reason. Do not relabel frozen earlier reports.

## Graph

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| G1 | Does each model component use an appropriate coarse semantic operation, rather than an unnecessarily expanded per-head/per-element graph? | Component-to-op map; construction/inference/evaluation contract; node count and compile cost. |
| G2 | Are proposed and retained fusion boundaries legal and useful? | Before/after producer-consumer chains; shared users, alias/effects, complete reductions, activation and rounding order, and actual fused implementation. Explain important unfused boundaries. |
| G3 | Are Pack, Slice, Cast, Concat and Unpack absent as standalone compute nodes after packing and after distribution? | Separate counts for each op, including numeric vectorized Cast, in entry, regions/loops and all reachable callees; exact checkpoint identities. Diagnose any remainder at its producer/consumer contract. |
| G4 | Are constant transforms actually proven pure and materialized offline? | Original constant/checkpoint provenance and readonly recipes; no runtime work hidden by a weights label. |
| G5 | Do graph edits preserve control flow, parameter banks, tuple/function ABI and state/effect ordering? | Complete-module before/after verification, call/callee mapping, loop yields, captures and repeated/sequential-call checks. |
| G6 | Does the change remove runtime work and improve the whole critical path? | FLOP/byte/recomputation/buffer accounting, counter-cost and a falsifiable whole-workload estimate; compilation time kept separate from measured runtime. |
| G7 | Does the implemented graph reach the best-known matched fusion/elimination or theoretical critical-path target? | Pinned external region/model baseline, complete fused-work boundary, lower-bound and emitted-graph accounting, measured result or proof, gap and next action. |

## Vector

**Check actual types, not merely vectorized op names or wide load instructions.**
In FlagMega a vector-valued tensor has a `TensorType` whose `dtype` is a
`VectorType(elem_type, lanes)`. Inspect the complete lane tuple and its scalar
coordinate map. A zero-Pack graph alone does not prove correct vectorization.
Inventory logical vector types at packed/vectorized and distributed checkpoints.
At final TIR, prove their lane-to-address mapping into physical buffers; a legal
lowering to scalar/byte storage does not by itself mean vectorization was lost.

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| V1 | Are tensors that should be vectorized represented by the correct `VectorType`, and are scalar tensors intentional? | Type inventory for activations, residuals, states, weights/results and function arguments/returns, including tuple fields. Record actual `tensor.dtype`, scalar element type, all lanes, outer shape and a decision for every retained scalar type. |
| V2 | Does packet geometry match the target and operand contract? | Lane-product, scalar extent and before/after byte-size calculations; multi-component lane-to-axis mapping. Widening `f16<8>` (8 scalars, 16 bytes) to `f32<2,4>` (8 scalars, 32 bytes) can preserve the required lane mapping while using two 16-byte output packets. Do not claim equal byte size or assume `f32<8>` has the same component-to-axis mapping. |
| V3 | Do producers, all consumers and every function boundary agree on representation? | Demand propagation through producers, residual/norm chains, calls/returns/tuples/regions; compatible ABI at every call site; no adapter merely moved or renamed. |
| V4 | Are indexing, permutation, padding and masks correct in scalar coordinates? | Before/after maps, lane order, tail/nondivisible/offset cases, logical versus physical strides and exact coverage; equal element counts are insufficient. |
| V5 | Are conversions and numeric boundaries preserved while representations change? | Scalar dtype and lane-aware PreOps/PostOps propagation, BF16/FP16 rounding and widening order, native quantized decode, cancellation/tie witnesses and evaluator/device distinction. |
| V6 | Do shard units and physical storage agree with the vector type? | Outer/lane split units, per-result byte spans, alignment, owners, aliases and caller/callee buffer ABI; coordinate proofs jointly checked with Distribution. |
| V7 | Does emitted code consume and produce the declared representation safely, with justified performance effects? | Actual buffer/operand types and load/store/indexing code, alignment and bounds; removed copies/buffers and issue/bandwidth/resource accounting. IR vector types do not require wide hardware loads when scalar access is the appropriate implementation. |
| V8 | Does the representation and access strategy reach the best-known matched or theoretical target? | Pinned target-specific layout/access baseline, actual VectorType/ABI and instructions, byte/transaction/issue costs, measured result or realizable-bound proof, gap and next action. |

Vector evidence includes this inventory schema, inline or as a linked private
JSON/CSV with a compact Markdown summary. Inventory all inspected tensor values;
group repeated equivalent rows only with explicit membership and counts.

| Function / value / tuple field | Role | Actual tensor.dtype | Scalar element / all lanes | Outer shape / scalar coordinate map | Bytes / strides / alignment / shard | Producer-consumer / function ABI | Decision / reason / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <stable value identity> | <activation/residual/state/weight/result/argument/return> | <VectorType or scalar DType> | <actual values> | <actual mapping> | <physical facts> | <all affected boundaries> | <vectorized correctly, repair, or justified scalar> |

Do not force every value into a vector type. Scalar indices/control values,
reduction/statistic outputs and fixed external interfaces can be appropriate;
state the reason and relevant consumer/target constraint. Unsupported packing
is a repair or explicitly justified retained representation, not an exemption
from the standalone-adapter acceptance gate.

## Distribution

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| D1 | Are mesh, placement, participants/tasks and B/S/P meanings correct for each tensor and operation? | Verify configuration through the actual public compiler entry: a CLI flag or a representable Placement alone does not prove the target accepts it. Inspect the configured target and distribution policy, then global-to-local coordinates and logical-to-program/CTA/warp/rank mapping, including vector split units, padded tails, replicas and partials. |
| D2 | Is distributed work balanced, with replication and idle participants justified? | Per-participant work/bytes, duplicated compute/requests, masks and critical-path estimate. Requested bytes are not automatically measured DRAM traffic. |
| D3 | Are reductions complete before operations that require complete values? | Partial/reduction ownership and communication graph; normalization/activation/rounding order; numerical and collective legality. |
| D4 | Are Boxing/collectives/transfers and publication necessary and implemented at the correct scope? | Producer-consumer dependencies, actual transfer/barrier/release-acquire path and cross-rank/CTA visibility. Explain retained or removed communication. |
| D5 | Does every ordinary physical store have valid ownership? | Disjoint or elected writers per scalar/result/state span, private replicas and tuple fields; declared atomic/reduction exceptions. Broadcast SBP does not permit overlapping canonical stores. |
| D6 | Are partition choices and scheduling/overlap real, legal and worthwhile? | Actual program boundaries, role lifetimes, blocking waits, load balance and demonstrated overlap; global producer roles must cover the outer model/layer execution when claimed. |
| D7 | Were dependent plans rebuilt consistently after a mapping change? | Complete distribution/selection/ABI/buffer/lifetime/synchronization/launch plan identities and numerical/device evidence; no stale counterpart plans. |
| D8 | Does the distribution reach the best-known matched placement/communication or theoretical critical-path target? | Pinned same-topology baseline, work/communication/rank bounds and actual schedule/overlap, measured result or achievable-model proof, gap and next action. |

## Kernel

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| K1 | Is the intended TIR/candidate/recipe/override applicable and actually selected? | Capability/type/shape/geometry/format guards, explicit registration path and identities, actual selected implementation; unsupported cases rejected or an explicitly declared ordinary path. When target configuration reconstructs a target, verify that installed local providers/recipes, implementation model, backend and unchanged policy options survive, while placement-dependent plans use the new configuration. |
| K2 | Does normal emitted code implement the complete numerical contract? | Decode, reduction/association, rounding/activation/math modes, masks/tails and per-result output dtype; CPU/source/NVCC/device scopes separate. |
| K3 | Are memory, address width, alignment and workspace contracts complete? | Actual buffers/aliases, strides/offsets/sizes, typed Shared/global resources, planner allocation/lifetime and physical ABI; no hidden allocation or generated-source patch. |
| K4 | Are stores, barriers, warp collectives and transfer lifetimes correct? | Physical writer census, uniform collective/barrier participation, publication/reuse/epoch proof, partial heads and empty/nonwriter/tail cases. |
| K5 | Are launch, warps/roles and resource use feasible and efficient? | Actual grid/block/warp allocations, registers/shared/local/stack/spills, residency and cooperative admission; distinguish static compilation from queried device attributes. Requested register caps or launch bounds do not prove residency; inspect spill costs and verify that the complete cooperative grid fits actual resident capacity. |
| K6 | Do indexing, loads/stores, tiling, vectorization and pipelines address the measured bottleneck? | Successful ncu on the actual baseline/candidate, emitted instruction/access pattern, WG/module traces and arithmetic/memory/issue ceilings; falsifiable estimates and counter-costs. No tensor-peak substitution for SIMT/integer paths. |
| K7 | Do partition and specialization lifetimes support the claimed prefetch/overlap? | Lowered outer schedule and role/wait/lifetime table, measured per-WG producer/consumer progress and hardware-module overlap with comparable clocks; fill/drain, waits, buffer epochs and bubbles. Function-local specialization that joins on return is not global model overlap; in-flight overlap is not measured engine-busy overlap. |
| K8 | Are resume, source/cache identity and reproduction complete? | Earliest affected stage, regenerated dependent plans and immutable source/recipe/IR/generated/library/manifest identities; fresh original-input plus public-package reconstruction and matching Verifier evidence. |
| K9 | Does the operator implementation improve a reused or traceably ported industry kernel and meet the declared target? Applicable industry code supplies the starting point; a citation or independent rewrite of the same idea does not establish reuse. A measured outperform objective requires a strict matched win; equality or theoretical attainment leaves it open. | Search date, primary-source URLs and exact implementation/benchmark; pinned upstream revision/license, concrete reused or ported code/mapping/memory strategy and required numerical/ABI adaptation diff; reproduced matched contract and raw measurements; applicable lower bound plus generated-code realization, target/gap and next action. |
| K10 | Is mandatory kernel microprofiling complete and independently supported? | ncu success/raw report and export, every WG in the declared capture scope, raw pipeline and hardware-module activity trace, correlated WG/module swimlane SVG, clock/coverage/drop/overhead checks and reproducible overlap analysis. For recorder changes, inspect the actual timestamp/write instructions, raw ABI/decoder and disabled/minimal/full perturbation controls using the [IKET-derived method](kernel-trace-method.md). Follow [Kernel microprofiling](kernel-microprofiling.md); ncu failures require sudo access or continued pause, never a profiler bypass. Missing hardware intervals keep this gate open. |

Verifier fixtures must satisfy the tested entrypoint's own pointer, allocation,
alignment and collective contracts. Wide-load fallback tests still need valid
scalar-aligned typed buffers. Use null only when the entrypoint forms no nonzero
pointer offset and performs no access; an empty inner loop alone is insufficient.
Record fixture or audit-harness failures separately from candidate numerical
failures, preserve their evidence, and repair the test before drawing a verdict.

Cross-domain findings return to the contract owner. Fusion does not belong only
to Graph, a vector packet does not determine sharding, and a legal mesh does
not prove safe physical stores. Joint Architect reports reconcile these checks;
Optimizer and Reviewer cover the same affected contracts on the actual revision.
