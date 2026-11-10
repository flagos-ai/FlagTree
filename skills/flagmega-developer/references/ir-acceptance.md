# IR acceptance and repair

Use this reference when reviewing an imported component, a new or extended
operation, a representation optimization, or a partition/scheduling change.
It defines acceptance criteria
and how to repair failures. A design-only request calls for a diagnosis and
repair design; these criteria do not authorize implementation beyond the
user's task.

## Acceptance scope and checkpoints

Inspect the typed Python module and its matching readable IR, including the
entry and every reachable reusable callee. Record input/compiler/target
identity, stage, semantic hash, and any local implementation overrides. Inspect
the complete affected producer/consumer chain, not just the newly added op.

Apply the structural gate after packing and function-layout propagation have
converged, then recheck after AutoDistributed and target-specific rewrites
before TIR lowering. Early imported IR, proposals, and intermediate rewrite
states may contain temporary Pack/Slice/Cast/Concat/Unpack nodes; they are not
the accepted optimized IR.
Reinspect lowered TIR, buffer plans and generated source to verify that the
accepted representation actually reached execution.

A stale dump from before a change cannot establish acceptance. Resume before
the earliest affected transformation and regenerate invalidated decisions;
follow [Performance iteration](performance-iteration.md). An op-level success
does not establish that its callee/caller chain or final artifact passed.

## Required checks

| Item | Acceptance criterion |
| --- | --- |
| Compute-body adapters | **Zero standalone Pack, Slice, Cast, Concat and Unpack nodes in compute bodies**, including the entry, loop/region bodies and every reachable callee. Check operation identities (`tensors.pack`, `tensors.slice`, `tensors.cast`, `tensors.concat`, `tensors.unpack`) and equivalent variants such as `ntt.vectorized_cast`. All five obey the same scope and acceptance rule; a zero-copy lowering does not exempt any of them. Required indexing, rounding and layout semantics belong in the owning producer/consumer contract. |
| Constants and weights | Weight transforms and constant splats have proven pure, deterministic, input/state-independent provenance and are evaluated/materialized offline through the normal constant mechanism. A displayed `weights` section alone proves neither folding nor execution placement. Runtime-derived values must remain in compute. |
| Producer/consumer representation | Producers emit the layouts consumers require. Verify every result field's scalar dtype, logical shape, full lane shape, lane-to-axis mapping, and divisibility/alignment; a packed weight does not imply a packed activation output. |
| Other representation boundaries | A retained byte view must have proven storage order, owner mapping and lifetime. Required numeric conversions must execute at their original semantic boundary inside the producer/consumer implementation; they do not justify standalone compute Cast/Unpack. Renaming a forbidden adapter to Bitcast or hiding it in a wrapper does not satisfy the gate. |
| Operation granularity | A component uses an operation that states its numerical contract at the appropriate granularity. No unnecessary per-head/per-stream expansion or graph-level unrolling of kernel-internal iterations. |
| Kernel/program partition and warp specialization | Analyze both kernel/program cuts and warp-specialization role partitions. Justify boundaries, role responsibilities and warp allocation by dependencies, target resources or measured performance. Verify the actual lowered `warp_specialize` scope: a global producer/consumer megakernel should cover outer model/layer execution. Assess every cut and blocking wait's effect on cross-layer weight prefetch, communication overlap and buffer lifetime. Unexplained per-op/per-layer splitting or unexamined loss of the intended overlap fails acceptance. |
| Numerical contract | Accumulation, reduction order, epsilon placement, intermediate rounding, and result dtype match the configured reference/acceptance policy. A layout change does not authorize precision changes. |
| Function ABI and shared users | Definitions, all call sites, tuple projections, sequential calls and public results agree on types and storage contracts. No duplicated expensive producer or repeated function bodies merely to satisfy layout demands. Propagation reaches a fixed point. |
| Distribution | Placement, split units, all lane components, partial reductions, Boxing and publication are correct. Layout optimization must preserve required communication; replication must follow algorithm/capability requirements. |
| Effects and physical storage | State reads/writes, aliases, owner mapping, allocation scope, scalar offsets, buffer lifetimes, scratch and synchronization are explicit and valid through lowering. |
| Executable coverage and evidence | Constructors, inference/evaluator, Python edit/resume, applicable candidates, renderers and selected device kernels implement the same contract. Preserve defect reproducer and fail-before/pass-after evidence; report untested/skipped scope separately. |

The no-Pack/Slice/Cast/Concat/Unpack gate is a structural requirement in addition
to numerical and runtime correctness. A clean dump cannot establish correctness
or speed, and a numerically correct or zero-copy adapter cannot waive the
structural requirement. Slice, Cast, Concat and Unpack have exactly the existing
Pack scope: compute bodies, including reachable callees. Pure offline
weight/constant recipes may use these ordinary operators
with proven constant provenance. Instructions inside an implementing device
kernel may perform the required addressing, conversion and rounding.
If a required producer/kernel capability is missing, record a failed item and
the causal repair needed; do not silently accept a reduced scope or disable
vectorization to pass the gate.

## Assess partitioning and warp-specialization scope

This static assessment is paired with the mandatory
[kernel microprofiling flow](kernel-microprofiling.md): ncu, actual per-WG
producer/consumer trace and hardware-module activity swimlanes. Use comparable
runtime intervals to establish overlap; lowered specialization alone is not
evidence of concurrent hardware execution.

Inspect partition metadata such as `kernel_program_partition`, the owning
pass/target decision, lowered IR containing the specialization regions,
generated entrypoints and the runtime launch graph. Distinguish kernel/program
cuts from role partitions within one kernel; assess both. Record both invocation
count and distinct compiled entry count; reusable code does not remove a kernel
boundary. Function names and source-level role labels do not prove the lowered
scope. Compare relevant tutorials or accepted baselines, but establish scope
from the current artifact.

For each retained boundary, establish:

- **Cause and justification:** identify the file/function/pass introducing the
  cut and the concrete data dependency, resource limit or parallel-work benefit
  it addresses. Automatically making every `tir.call` a program is not itself
  a justification. Compare the proposed cuts with an outer model execution.
- **Role scope and progress:** trace the complete layer sequence in producer
  and consumer. A global `warp_specialize` around that sequence lets producer
  warps naturally advance across layers; it does not inherently require an
  additional cross-layer scheduler. A `warp_specialize` inside a reusable
  function that joins before returning does not provide this global role
  lifetime, even when successive calls use the same producer label. Verify
  that the outer specialization encloses the layer/call sequence in lowered
  IR. Locate any per-layer specialization join,
  function return, kernel retirement, handoff, pipe acquire/drain or barrier
  that prevents this progress.
- **Role assignment and warp allocation:** identify producer, consumer and any
  communication role, their work and warp/thread ranges in the lowered IR.
  Explain why the split supplies the transfer, compute and communication work
  without avoidable serialization or idle warps. Account for each role's
  register needs and shared-buffer ownership as well as total kernel residency;
  fewer warps or an extra communication partition is not automatically better.
- **Prefetch and communication dependencies:** in a producer/consumer pipeline,
  keep readonly weight prefetch in the producer. Distinguish readonly L2 hints
  from TMA into shared pipe slots, current-tile lookahead from next-layer weight
  prefetch, and immutable sources from runtime-selected weights. Prove source
  availability and shared-slot reuse. Trace shared-buffer aliases and lifetimes
  across caller/callee boundaries: consumer scratch may prevent the producer
  from loading the next layer even when its readonly source is ready. Identify
  the protected storage before deciding whether a handoff or wait is avoidable;
  preserve necessary synchronization and repair the allocation/lifetime plan
  when it causes unwanted serialization. An early L2 hint does not establish
  overlap of the actual shared-memory transfer.
  Identify which role publishes, receives
  and waits for communication; justify a blocking communication wait in the
  producer with a real dependency on its next transfer. Assess overlap with
  both producer prefetch and independent consumer work, and identify the data
  dependency that ends each overlap window; do not assume either is sufficient
  from role placement alone.
  In particular, when next-layer weights and destination capacity are already
  available, explain any communication wait that stalls their producer even
  though the transfer does not consume the communication result. Evaluate
  placing the wait at the actual dependency boundary or using a communication
  role, preserving collective participation and buffer publication. Do not
  require a separate communication partition when the current roles provide
  the justified overlap.
- **Correctness and cost:** preserve numerical order, publication/visibility
  and buffer lifetimes across the proposed boundaries. Compare launch and
  synchronization costs, concurrent work, register/shared-memory use and
  residency. Claims of beneficial overlap require full requested-workload
  measurements or profiler evidence; static role placement alone is insufficient.

If a cut turns an intended global producer/consumer sequence into isolated
per-op or per-layer regions without an established need and an assessed overlap
tradeoff, leave this acceptance item failed. Repair the partition/scope decision
and its dependency/lifetime contracts through normal lowering; do not accept it
merely because compilation or numerical checks pass, patch only the generated
source, or remove necessary synchronization to make overlap appear.

Report a table for the assessed cuts and role partitions: decision/owner,
actual scope and work, warp allocation, blocking dependency, cross-layer
prefetch window, overlap with producer and consumer, resource/performance
evidence, and the retain/merge/redesign conclusion. Mark missing evidence
explicitly. Use the requested execution phase for performance acceptance;
decode-only tuning requires decode evidence rather than prefill measurements.
A justified partition remains allowed; an unmeasured performance claim remains
unproven.

## Find every remaining forbidden compute adapter

Inspect graph nodes by operation identity, including vectorized conversion
variants, rather than matching names ending in `pack0` or trusting pretty-print
grouping. Count Pack, Slice, Cast, Concat and Unpack separately. Use the actual
function scopes and constant provenance to distinguish compute nodes from
constant recipes. A whole-file keyword count in `.il` is not a compute-body
acceptance count: classify each occurrence by execution scope and provenance.
Keep ordinary Pack expressions in the constant mechanism; do not introduce a
parallel weight-only op family or move an activation into `weights` by editing
printer metadata.

For each forbidden compute node, record:

- Function, operation identity, node, producer, and all consumers, including
  exported values and tuple/call boundaries.
- Input/output scalar dtype, logical shape, lane components and producer result
  field; relevant axes, selection ranges/steps, concatenation order or conversion
  sequence and rounding.
- The transformation that inserted or exposed it, and the point where demand
  propagation stopped.
- Missing producer/type/kernel support, unsupported axis mapping, or conflicting
  layout decisions; assign the fix to that layer.

Count all affected functions, not one favorable example. Deduplicating two
identical adapters into one does not resolve the failed gate. A zero-Pack report
alone no longer establishes structural acceptance.

## Repair the cause

| Cause | Repair |
| --- | --- |
| Producer always returns a scalar tensor | Extend its output representation contract, preferably through an existing layout attribute or an explicit optional per-result lane attribute. Implement shape/dtype inference, evaluator scalar order and rounding, API/serialization, distribution and actual kernel output addressing together. Keep scalar default compatibility. |
| Producer supports the layout but no rule reaches it | Add consumer-demand propagation at the existing packing stage that owns this decision. Handle `GetItem`, tuple and reusable-call boundaries; rebuild a shared producer once and preserve other result fields and consumers. Register the rule on all applicable paths, including ordinary and large-graph paths. |
| Demand stops at reshape/slice/transpose | Prove the axis mapping and lane order through the transform. For slices, check starts, steps, extents and alignment. Propagate only the legal layouts; equal element counts alone are insufficient. |
| Function boundary restores the old layout | Reconcile parameter and result ABIs across every call site, then rerun producer demand and inverse-view folding to a fixed point. Preserve public results through producer/consumer layout support without reintroducing any forbidden compute adapter. |
| Kernel/program cuts truncate global producer/consumer progress | Repair the owning partition and role-scope plan so specialization covers the intended outer execution. Reestablish source-readiness, pipe-slot reuse, shared-buffer lifetimes and communication ordering; retain only justified boundaries and validate full-workload correctness, resources and performance. |
| Different consumers choose incompatible layouts | Choose a coherent legal producer/consumer plan from actual demand and target contracts. Preserve shared scalar or other-layout users using proven views where possible; update downstream contracts where needed. Do not clone expensive computation or force one global vector width. |
| An inverse boundary fails to fold | Repair the generic equality/view proof, including normalized negative axes, repeated axes, partial unpacking and all lane components. Check that result type and storage order match; add negative cases for transforms that genuinely permute data. |
| Pure indexing materializes a private consumer input | Compose its coordinate map with consumer loads, or use a proven storage view. For coordinated state writes, express the complete semantic update and its epoch/barrier contract. Verify that temporary buffers and copy calls disappear; see [Consumer indexing fusion](consumer-indexing-fusion.md). |
| Slice remains between producers and consumers | Fuse starts/steps/extents and masks into consumer addressing, or make the producer emit the selected result field directly. Prove packed-lane alignment, physical strides, ownership and bounds. For shared/exported selections, repair every user's input contract or the producer's result ABI; retaining a standalone Slice is a failed gate. |
| Cast carries required rounding | Fuse the exact conversion sequence into a producer epilogue or consumer prologue supported by inference, evaluator, serialization and the selected kernel. Keep accumulation dtype, intermediate rounding and conversion order explicit. Remove the rounding itself only when the agreed numerical policy permits it; never move it across a reduction, collective or shared use without proof. |
| Concat assembles partitions only to feed another op | Let the consumer accept the original parts with an explicit axis/order/extent contract, or have producers write proven disjoint regions of the final result. A partitioned projection/reduction should express its rounded partials and reduction order directly. Preserve shared users, aliases and publication; a renamed Concat or opaque wrapper is not fusion. |
| Unpack restores scalar or another lane layout | Propagate a coherent representation through the consumer, tuple projections and all call sites; let the kernel address scalar lanes of the packed source directly. Prove full lane order and physical ABI, preserve other users, and fold inverse views to a fixed point. Do not duplicate the producer or merely rename Unpack to Bitcast. |
| A constant adapter appears in compute | Prove its dependencies are pure, deterministic and invariant, then fix ordinary constant propagation/recipe construction so it is evaluated offline. Mutable state, tokens, and runtime-derived values cannot be moved to constants. |
| Kernel rejects or misaddresses packed output | Update candidate legality and renderer/template addressing using the physical ABI's logical group and scalar lane coordinates. Validate full lane shape, scalar dtype, capacity, strides, offsets and owners. Dropping a scalar-only guard without implementing addressing is insufficient. |

Do not pass the gate by deleting a required adapter's semantics, replacing it
with an unproved Bitcast, editing only printed types, hiding it in a dump,
disabling a pass, or choosing
scalar compute simply to avoid exposing missing packed support. For a real
layout permutation, design native producer/consumer layout support or a
semantically justified transformation; replacing the op name alone is not a
repair. Until that contract is implemented and validated, the item remains
failed. See [IR extension SOP](ir-extension-sop.md) for the complete support
path and [Representation optimization](representation-optimization.md) for
numerical and view proofs.

## Examples from decoder layout integration

**MhcPre:** `MhcPre -> GetItem(2) -> Pack -> NormStats/NormApply` requires the
collapsed field to be produced in the demanded layout. Keep `post`/`comb` FP32,
preserve collapse rounding, propagate through tuple users, and implement
packed output addressing. Packing coefficients or every tuple field is not
the same fix.

**FP8 MLA projections:** `PackedBlockScaledMatMul -> Pack -> normalization`
or `Pack -> query reshape -> attention` still fails the gate. The existing
`Packed` name describes weight K packing and does not establish packed result
support. Repair output layout inference/evaluation, N-axis distribution split
units, consumer propagation and kernel output ABI; preserve FP8 scale/block
semantics and required final rounding. Packing width comes from downstream
demand and capability, not a special case for `decode_mla_attention_sparse`.

**KDA activation:** `Kda[Gated]RecurrentCore -> GetItem(0) -> Pack` requires
packed activation output support. Keep tuple field 1 scalar FP32, retain one
stateful producer at its original position, and preserve sequential state
users. Kernel head offsets must use each buffer's physical owner mapping and
scalar strides; multiplying a local head index by a stride alone cannot address
a canonical external buffer under head sharding. Test both contiguous and
block-cyclic ownership, including active heads at a padded tail.

**Stream reduction:** `ReduceSum -> Pack -> vectorized mean/normalization`
requires packed result stores while the reduction still accumulates the original
scalar domain in FP32 and rounds once to the declared scalar dtype. Scale split
units on the surviving packed axis without changing the reduced partial mesh
axes. Do not move packing onto the reduction input without proving its axes
and arithmetic contract.

**KV split before cache writes:** a combined per-head `[K | V]` projection
followed by two Slices and two cache updates can expose redundant intermediate
copies. A proposed paired cache operation can read both fields directly from
the combined tensor, preserve producer rounding and required Boxing, and
advance the shared sequence only after both writes complete. Inspect buffers
and selected source; graph adjacency alone does not prove zero-copy behavior.
This example does not establish that a particular checkpoint passes. Standalone
compute Slices fail the structural gate even when their selection is necessary;
preserve that selection in the owning contract. Follow
[Consumer indexing fusion](consumer-indexing-fusion.md).

**Weight Pack:** a pure checkpoint-weight transform or constant-zero Pack can
remain in a proven compile-time recipe with offline materialization. It must
not execute on each decode invocation. This allowance does not apply to an
activation merely because a later Pack lowering would be zero-copy.
The same provenance-based allowance applies to offline weight Slice, Cast,
Concat and Unpack; it does not exempt their runtime-derived counterparts.

## Acceptance evidence and reporting

Retain the inspected checkpoints and assert the structural gate on typed IR.
Then validate affected rules/passes, mixed/shared users, tuple/function ABI,
distribution, edit/resume compatibility, and the selected device kernels.
Check final buffers/source for unexpected materialization, copies, duplicated
producers or kernels. Measure compile time and runtime separately when claiming
an improvement.

Report checkpoint identity and separate compute Pack/Slice/Cast/Concat/Unpack
counts per affected function (all required to be zero), remaining compile-time
transforms and their provenance, numerical/device
results, neighboring scope checks, and any failed or unvalidated items.
Include partition justification, actual `warp_specialize` scope, effects on
cross-layer prefetch and communication overlap, and the evidence supporting
retained boundaries.
The current compiler is not assumed to satisfy the policy merely because the
policy has been documented. Leave a failed gate explicit while investigating
the causal implementation; do not relabel a diagnostic checkpoint as accepted.
