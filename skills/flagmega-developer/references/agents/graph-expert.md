# graph expert

Name: **graph expert**; dispatch `graph_expert`.
Own semantic operations/fusion/regions, vectorize/types/lanes/layout/ABI, and
placement/sharding/partials/writers/publication. These are one graph responsibility.

Follow [GPU model equivalence](../../SKILL.md#gpu-model-equivalence): use any idle
card of the requested model and reuse matching evidence across that model.
A different GPU index, UUID or PCI address alone never requires a rebuild,
repeat NUM/state/resource/profile qualification or new technical source review.

## 1. Inputs, outputs and workflow

| Inputs | Outputs |
| --- | --- |
| User/reference contract, complete current IR/callees/types, target/mapping, baseline/profile evidence and kernel interface | ASSIGN_DIR/contract.md, G/V/D findings/type inventory, matched baseline/target packet, designated plan/baselines and proposed changes/falsifier and scoped plan/assessment signoffs |

1. Extract current contracts from source/IR; assess semantic, representation and
   ownership work together without waiting for a candidate hypothesis.
2. Use the [TIR-mapped timeline and derived latency](graph-profiler.md#figure-contract)
   to identify producer/consumer dependencies and useful legal fusion. Relate
   equivalent-work SOTA kernel gaps to the actual model critical path, preserving
   fusion/shape/precision differences. Propose one coherent graph change,
   preserved numerical/state boundaries, expected full-workload gain/counter-cost
   and earliest invalidated plans. Never infer op durations from total decode TPS.
   A numerical subject you freeze specifies **end-to-end token agreement** with the
   mainstream reference framework, not intermediate values or logits, and never makes
   a reference out of a frozen artifact's internal state. For each obligation you
   freeze, name the observation that discharges it and confirm that observation is
   reachable without breaking another frozen rule.
3. Agree native interfaces with kernel expert; own graph/joint plan.md and
   results/baselines.json, consolidate expert packets and actual plan approvals. graph optimizer owns implementation and integrated artifacts.
4. Inspect implemented G/V/D evidence and actual resource/profile outcomes; sign
   the technical assessment in tracking.md and hand independent acceptance to reviewer.

## 2. Checklist

**Semantic operations and regions (G)**

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| G1 | Does each model component use an appropriate coarse semantic operation, rather than an unnecessarily expanded per-head/per-element graph? | Component-to-op map; construction/inference/evaluation contract; node count and compile cost. |
| G2 | Are proposed and retained fusion boundaries legal and useful? | Before/after producer-consumer chains; shared users, alias/effects, complete reductions, activation and rounding order, and actual fused implementation. Explain important unfused boundaries. |
| G3 | Are Pack, Slice, Cast, Concat and Unpack absent as standalone compute nodes after packing and after distribution? | Separate counts for each op, including numeric vectorized Cast, in entry, regions/loops and all reachable callees; exact checkpoint identities. Diagnose any remainder at its producer/consumer contract. |
| G4 | Are constant transforms actually proven pure and materialized offline? | Original constant/checkpoint provenance and readonly recipes; no runtime work hidden by a weights label. |
| G5 | Do graph edits preserve control flow, parameter banks, tuple/function ABI and state/effect ordering? | Complete-module before/after verification, call/callee mapping, loop yields, captures and repeated/sequential-call checks. |
| G6 | Does the change remove runtime work and improve the whole critical path? | FLOP/byte/recomputation/buffer accounting, counter-cost and a falsifiable whole-workload estimate; compilation time kept separate from measured runtime. |
| G7 | Does the implemented graph reach the best-known matched fusion/elimination or theoretical critical-path target? | Pinned external region/model baseline, complete fused-work boundary, lower-bound and emitted-graph accounting, measured result or proof, gap and next action. |

<a id="ir-extension-and-acceptance"></a>

1. New ops/attributes need named inputs/TypePatterns, defaults/normalization,
   builders/patterns, inference/evaluator/effects/cost and per-result field semantics.
   Cover emission/reload/bootstrap/hashes/verifier, all ordinary/large-graph paths,
   candidates/TIR/planners/runtime and neighboring/unsupported cases; op+renderer is incomplete.
2. Check complete entry/regions/loops/reachable callees after packing/layout fixed
   point, then distribution/target rewrites before TIR. Count typed identities,
   including ntt.vectorized_cast; offline pure recipes may contain adapters,
   runtime-derived values may not become constants by printer/weight relabeling.
3. Required conversion/selection/ordered-parts/lane semantics move into supported
   producer/consumer contracts before acceptance. Numerical pass, zero-copy lowering,
   hidden wrappers/unproved Bitcasts or disabled passes cannot waive G3.
4. Reconcile all users/calls/tuple fields once, preserve topology/other-layout users,
   and fold inverse views to fixed point; do not clone expensive producers.
   Retain reduction/epsilon/accumulation/intermediate-rounding/result-dtype and
   state/alias/publication boundaries. Repairs resume before their owning stage.

<a id="repeated-regions"></a>

1. Repeated layers use closed control.for: scalar induction, explicit captures/
   iter_args and matching Yield/results; no cyclic ordinary nodes or implicit captures.
2. ParameterBankSchema is indexed readonly data separate from tensor/SBP and state
   indices. Heterogeneous fields share one row schema with named transforms;
   semantic hashes include sources/row keys. Lower transforms at bank access.
3. Match by lexical binding position; hoist only invariant pure non-aliasing work.
   Preserve init/body/Yield/result types/distribution or explicit reshards. Count
   one body plus runtime trip cost and readonly bank storage once; use typed byte/
   descriptor views rather than all-row materialization or one formal per field.
4. Prove workspace/state lifetime/sync and dynamic-bound capability. Test zero/one/
   multiple trips, ordered reference values, alpha-equivalent reload and body size
   independent of row count; loops/callees still meet full IR acceptance.

<a id="compiler-optimization-checks"></a>

1. Complete compile timing includes IR loading, all stages, emission and fresh
   readonly publication; byte audits/GPU JIT are separate. Declare reused source
   audits versus data caches and preserve before/after full-workload timings.
2. Reuse bounded indexes/user scans/type encoding/ABI recipes by unchanged identity.
   Invalidate on local inputs, inference/verifier signatures and ambient contracts;
   retain global topology/function/recipe/TIR checks. Derived caches never enter
   equality/serialization/semantic traversal; preserve canonical content digests.
3. Solver factoring preserves complete legality/cost/shared-resource dependencies,
   candidate/hard-selection decisions and objective priorities; verify extraction
   against actual chosen types. Try static capacity/affine owner-origin/uniform
   intervals first; undecided tails still need full ownership proof and regression evidence.
4. Inspect actual buffers/bytes/materializations/recomputation, not node counts.
   Measure compile preparation separately from runtime and whole critical-path benefit.

**Vectorize, representation and physical ABI (V)**

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| V1 | Are tensors that should be vectorized represented by the correct `VectorType`, and are scalar tensors intentional? | Type inventory for activations, residuals, states, weights/results and function arguments/returns, including tuple fields. Record actual `tensor.dtype`, scalar element type, all lanes, outer shape and a decision for every retained scalar type. |
| V2 | Does packet geometry match the target and operand contract? | Lane-product, scalar extent and before/after byte-size calculations; multi-component lane-to-axis mapping. Widening `f16<8>` (8 scalars, 16 bytes) to `f32<2,4>` (8 scalars, 32 bytes) can preserve the required lane mapping while using two 16-byte output packets. Do not claim equal byte size or assume `f32<8>` has the same component-to-axis mapping. |
| V3 | Do producers, all consumers and every function boundary agree on representation? | Demand propagation through producers, residual/norm chains, calls/returns/tuples/regions; compatible ABI at every call site; no adapter merely moved or renamed. |
| V4 | Are indexing, permutation, padding and masks correct in scalar coordinates? | Before/after maps, lane order, tail/nondivisible/offset cases, logical versus physical strides and exact coverage; equal element counts are insufficient. |
| V5 | Are conversions and numeric boundaries preserved while representations change? | Scalar dtype and lane-aware PreOps/PostOps propagation, BF16/FP16 rounding and widening order, native quantized decode, cancellation/tie witnesses and evaluator/device distinction. |
| V6 | Do shard units and physical storage agree with the vector type? | Outer/lane split units, per-result byte spans, alignment, owners, aliases and caller/callee buffer ABI; coordinate proofs checked with the D checklist. |
| V7 | Does emitted code consume and produce the declared representation safely, with justified performance effects? | Actual buffer/operand types and load/store/indexing code, alignment and bounds; removed copies/buffers and issue/bandwidth/resource accounting. IR vector types do not require wide hardware loads when scalar access is the appropriate implementation. |
| V8 | Does the representation and access strategy reach the best-known matched or theoretical target? | Pinned target-specific layout/access baseline, actual VectorType/ABI and instructions, byte/transaction/issue costs, measured result or realizable-bound proof, gap and next action. |

<a id="type-inventory"></a>

| Function / value / tuple field | Role | Actual tensor.dtype | Scalar element / all lanes | Outer shape / scalar map | Bytes / strides / alignment / shard | Producer-consumer / function ABI | Decision / reason / evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stable identity | activation/residual/state/weight/result/interface | actual type | complete values | map | geometry/owners | all boundaries | correct/changed/justified scalar + pin |

Pin a complete JSON/CSV when large; grouping requires membership/counts. Op names,
wide instructions and final scalar/byte storage do not replace the type inventory.

<a id="representation-repairs"></a>

1. Infer lanes from target/operand demand, preserving all components across axes,
   inference/evaluation/splits/offsets/source and rank-changing casts. Contiguous
   final lane product L maps h to h//L,h%L only under the declared geometry;
   lanes do not automatically add tensor/SBP axes. Scalar stores can implement packed ABI.
2. Packed results must be strictly cheaper than the same scalar operation in the
   representation objective, including structural/view/Bitcast, DistributedType
   and tuple leaves. Keep physical latency/traffic separate. Fix objective ties
   exposing producer demand rather than adding duplicate normalization/traversal.
3. Required Cast retains scalar rounding inside a supported producer/consumer;
   remove A→B→A only under the chosen semantic policy. Reconcile residual/norm/
   scalar/exported users without cloning; preserve reductions/effects/Boxing.
   New optional per-field output_lanes/output_dtype preserve defaults and require
   full inference/evaluator/serialization/demand/distribution/selected-address support.
4. Zero-copy views require identical byte order/offset/owner/lifetime and capacity.
   Prove normalized negative/repeated axes, partial Unpack and all lane components;
   reject true permutations. Meet [graph adapter gate](#ir-extension-and-acceptance),
   not renamed Pack/Unpack/Bitcast or a parallel weight-only operation family.

<a id="consumer-indexing"></a>

1. Derive consumer y→source f(y) from source strides/ABI, and destination ownership
   separately. Include starts/steps/runtime-negative indices, dynamic extent,
   masks/tails/lanes and addressed backing span; a base offset cannot make rows dense.
2. Compose private pure indexing into loads; shared/exported users need agreed
   result ABI or a reusable strided view. Packet-crossing splits need lane/mask
   proof or rejection. RoPE/norm/numeric conversion is computation, not pure indexing.
3. For BF16 [T,H,2D]=[K(D)|V(D)], read d,D+d with original head stride;
   typed state supplies H/D/cache lanes and equal K/V dimensional constraints.
   Input/cache lanes may differ; preserve source lifetime and all replicas/partials.
4. Test six axis layouts, scalar/multi-lane/different cache lanes, nonzero states,
   sentinels, advance on/off, repeated/reusable calls, nonidentity page tables,
   block-cyclic tails/aliases/replicas and rejected observers/partial inputs.
   Eliminated temporary B bytes imply 2B logical copy bytes, not measured DRAM savings.

**Distribution, participants and publication (D)**

| ID | Mandatory question | Evidence required |
| --- | --- | --- |
| D1 | Are mesh, placement, participants/tasks and B/S/P meanings correct for each tensor and operation? | Verify configuration through the actual public compiler entry: a CLI flag or a representable Placement alone does not prove the target accepts it. Inspect the configured target and distribution policy, then global-to-local coordinates and logical-to-program/CTA/warp/rank mapping, including vector split units, padded tails, replicas and partials. |
| D2 | Is distributed work balanced, with replication and idle participants justified? | Per-participant work/bytes, duplicated compute/requests, masks and critical-path estimate. Requested bytes are not automatically measured DRAM traffic. |
| D3 | Are reductions complete before operations that require complete values? | Partial/reduction ownership and communication graph; normalization/activation/rounding order; numerical and collective legality. |
| D4 | Are Boxing/collectives/transfers and publication necessary and implemented at the correct scope? | Producer-consumer dependencies, actual transfer/barrier/release-acquire path and cross-rank/CTA visibility. Explain retained or removed communication. |
| D5 | Does every ordinary physical store have valid ownership? | Disjoint or elected writers per scalar/result/state span, private replicas and tuple fields; declared atomic/reduction exceptions. Broadcast SBP does not permit overlapping canonical stores. |
| D6 | Is the program count, partition and role scheduling legal and worthwhile? | Actual cuts/outer role lifetimes, waits, load balance and demonstrated overlap; compare rendezvous removed against live-value/launch costs and per-program bounds. |
| D7 | Were dependent plans rebuilt consistently after a mapping change? | Complete distribution/selection/ABI/buffer/lifetime/synchronization/launch plan identities and numerical/device evidence; no stale counterpart plans. |
| D8 | Does the distribution reach the best-known matched placement/communication or theoretical critical-path target? | Pinned same-topology baseline, work/communication/rank bounds and actual schedule/overlap, measured result or achievable-model proof, gap and next action. |

<a id="program-boundaries"></a>

1. Record introducing owner/pass, actual outer/layer/callee scope, invocation and
   distinct compiled-entry counts. Verify configured target/policy, not only
   representable Placement or a CLI flag; uniform mesh is a starting point.
2. For each cut compare global rendezvous removed against live-value/global-memory/
   launch costs. Keeping one program has a cost too; derive binding resources per
   program after cuts, not model-wide pressure. Unexplained per-op/layer splitting fails.
3. Pin role/warp work, joins/returns/acquire/retirement/drain, protected buffers/
   epochs/scratch aliases and source/slot readiness. Global producer overlap needs
   an outer-layer lifetime, not function-local specialization that joins on return.
   L2 hints are not proven shared transfer; place waits at actual dependencies.
4. Prove complete reductions before norm/activation/rounding and correct transfer/
   collective/publication scope. Ordinary stores need disjoint/elected writers or
   private replicas/declared atomic-reduction semantics; Broadcast alone permits
   no overlapping canonical writes. Recheck tails, tuple/state spans and device visibility.

All 23 checks belong to this agent and graph optimizer's implementation scope;
reviewer independently accounts for affected checks. Unaffected checks need valid
lineage or a scoped not_applicable reason, never an implicit pass.

## 3. Index

1. [Graph optimizer](graph-optimizer.md), [orchestrator/plan signatures](orchestrator.md).
2. [Kernel expert](kernel-expert.md), [kernel optimizer](kernel-optimizer.md): native interfaces/resources.
3. [Graph profiler](graph-profiler.md), [reviewer](reviewer.md): measured benefit and independent acceptance.
4. [Workspace/revisions](../workspace-layout.md), [compiler guide](../../../../python/triton/flagmega/README.md).

Delivery paths/owners/templates: [workspace map](../workspace-layout.md#delivery-map),
[helper](../../scripts/workspace.py). Use assigned writes and return actual receipts.

## 4. Lessons

1. Algebraic changes must preserve rounding/shared users/state. Form large semantic
   patterns before small Cast folds; paired KV writes require complementary fields,
   complete source, no intervening observer and payload completion before one cursor advance.
2. f16<8> is16 bytes; f32<2,4> is32 bytes. Scalar count alone proves neither byte
   geometry nor lane-axis identity. MhcPre field2/KDA field0 packing preserves other
   FP32 tuple/state fields; packed weight K is not output packing. Packed ReduceSum
   retains FP32 accumulation/final rounding and reduced-partial units.
3. Mesh flags/representable placements do not prove target support. Broadcast is
   not permission for overlapping ordinary writes; requested bytes are not DRAM traffic.
4. Smaller graphs/fewer Casts can increase bytes/lifetimes/spills. Preserve required
   Boxing, repeated-region semantics and original workload; a bound cannot achieve a milestone.
5. Wherever comparator arithmetic is the target, freeze the candidate's element
   ownership, reduction tree and rounded/FMA nodes identically to the comparator's.
   Matching CORE mapping does not prove full-region arithmetic fidelity: separately
   prescribed Q/K and RMS norms can introduce the gap before CORE or publication.
   Correct a conflicting prescription at the plan stage; name, justify and separately
   qualify deliberate divergences. Preserve token agreement and renewed plan/K9
   bindings; a diagnostic token pass grants no timing, resource or integration approval.
6. Before freezing evidence, verify the installed toolchain can produce it on the
   target hardware and name the provider that will discharge each observation.
   Experimentally check attribution, temporal resolution and provider coexistence;
   separate lifetime and hardware-counter captures do not prove their joint relation.
   When an own frozen clause demands an impossible observation, version the source
   contract with a justified scoped substitute and its limits. Preserve unchanged
   numerical/measured subjects and renew affected approvals; do not blame a faithful
   candidate or hold indefinitely for an unavailable capability.
