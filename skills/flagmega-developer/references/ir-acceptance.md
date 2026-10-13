# IR acceptance and repair

Use this reference when reviewing an imported component, a new or extended
operation, or a representation optimization. It defines acceptance criteria
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
states may contain temporary Packs; they are not the accepted optimized IR.
Reinspect lowered TIR, buffer plans and generated source to verify that the
accepted representation actually reached execution.

A stale dump from before a change cannot establish acceptance. Resume before
the earliest affected transformation and regenerate invalidated decisions;
follow [Performance iteration](performance-iteration.md). An op-level success
does not establish that its callee/caller chain or final artifact passed.

## Required checks

| Item | Acceptance criterion |
| --- | --- |
| Compute-body Pack | **Zero `tensors.pack` / `Pack` nodes in compute bodies**, including reachable callees. A producer's scalar output followed by Pack is incomplete layout support, even if bufferization later turns Pack into a zero-copy view. |
| Constants and weights | Weight transforms and constant splats have proven pure, deterministic, input/state-independent provenance and are evaluated/materialized offline through the normal constant mechanism. A displayed `weights` section alone proves neither folding nor execution placement. Runtime-derived values must remain in compute. |
| Producer/consumer representation | Producers emit the layouts consumers require. Verify every result field's scalar dtype, logical shape, full lane shape, lane-to-axis mapping, and divisibility/alignment; a packed weight does not imply a packed activation output. |
| Other representation boundaries | Remove redundant inverse views and repeated conversions where proven legal. Retained Bitcast/Unpack/Cast operations must express a required ABI, byte view, or rounding boundary; preserve required numeric conversions. Renaming Pack to a view does not satisfy producer layout support. |
| Operation granularity | A component uses an operation that states its numerical contract at the appropriate granularity. No unnecessary per-head/per-stream expansion or graph-level unrolling of kernel-internal iterations. |
| Numerical contract | Accumulation, reduction order, epsilon placement, intermediate rounding, and result dtype match the configured reference/acceptance policy. A layout change does not authorize precision changes. |
| Function ABI and shared users | Definitions, all call sites, tuple projections, sequential calls and public results agree on types and storage contracts. No duplicated expensive producer or repeated function bodies merely to satisfy layout demands. Propagation reaches a fixed point. |
| Distribution | Placement, split units, all lane components, partial reductions, Boxing and publication are correct. Layout optimization must preserve required communication; replication must follow algorithm/capability requirements. |
| Effects and physical storage | State reads/writes, aliases, owner mapping, allocation scope, scalar offsets, buffer lifetimes, scratch and synchronization are explicit and valid through lowering. |
| Executable coverage and evidence | Constructors, inference/evaluator, Python edit/resume, applicable candidates, renderers and selected device kernels implement the same contract. Preserve defect reproducer and fail-before/pass-after evidence; report untested/skipped scope separately. |

The no-Pack gate is a structural requirement in addition to numerical and
runtime correctness. A clean dump cannot establish correctness or speed, and
a numerically correct zero-copy Pack cannot waive the structural requirement.
If a required producer/kernel capability is missing, record a failed item and
the causal repair needed; do not silently accept a reduced scope or disable
vectorization to pass the gate.

## Find every remaining Pack

Inspect graph nodes by operation identity rather than matching names ending in
`pack0` or trusting pretty-print grouping. Use the actual function scopes and
constant provenance to distinguish compute nodes from constant recipes. Keep
ordinary Pack expressions in the constant mechanism; do not introduce a
parallel weight-only op family or move an activation into `weights` by editing
printer metadata.

For each compute Pack, record:

- Function, Pack node, producer, and all consumers, including exported values
  and tuple/call boundaries.
- Input/output scalar dtype, logical shape, lane components, normalized pack
  axes and mapping to the producer's result field.
- The transformation that inserted or exposed it, and the point where demand
  propagation stopped.
- Missing producer/type/kernel support, unsupported axis mapping, or conflicting
  layout decisions; assign the fix to that layer.

Count all affected functions, not one favorable example. Deduplicating two
identical activation Packs into one does not resolve the failed gate.

## Repair the cause

| Cause | Repair |
| --- | --- |
| Producer always returns a scalar tensor | Extend its output representation contract, preferably through an existing layout attribute or an explicit optional per-result lane attribute. Implement shape/dtype inference, evaluator scalar order and rounding, API/serialization, distribution and actual kernel output addressing together. Keep scalar default compatibility. |
| Producer supports the layout but no rule reaches it | Add consumer-demand propagation at the existing packing stage that owns this decision. Handle `GetItem`, tuple and reusable-call boundaries; rebuild a shared producer once and preserve other result fields and consumers. Register the rule on all applicable paths, including ordinary and large-graph paths. |
| Demand stops at reshape/slice/transpose | Prove the axis mapping and lane order through the transform. For slices, check starts, steps, extents and alignment. Propagate only the legal layouts; equal element counts alone are insufficient. |
| Function boundary restores the old layout | Reconcile parameter and result ABIs across every call site, then rerun producer demand and inverse-view folding to a fixed point. Preserve public results with proven ABI adapters without reintroducing compute Pack. |
| Different consumers choose incompatible layouts | Choose a coherent legal producer/consumer plan from actual demand and target contracts. Preserve shared scalar or other-layout users using proven views where possible; update downstream contracts where needed. Do not clone expensive computation or force one global vector width. |
| An inverse boundary fails to fold | Repair the generic equality/view proof, including normalized negative axes, repeated axes, partial unpacking and all lane components. Check that result type and storage order match; add negative cases for transforms that genuinely permute data. |
| Pure indexing materializes a private consumer input | Compose its coordinate map with consumer loads, or use a proven storage view. For coordinated state writes, express the complete semantic update and its epoch/barrier contract. Verify that temporary buffers and copy calls disappear; see [Consumer indexing fusion](consumer-indexing-fusion.md). |
| Constant Pack appears in compute | Prove its dependencies are pure, deterministic and invariant, then fix ordinary constant propagation/recipe construction so it is evaluated offline. Mutable state, tokens, and runtime-derived values cannot be moved to constants. |
| Kernel rejects or misaddresses packed output | Update candidate legality and renderer/template addressing using the physical ABI's logical group and scalar lane coordinates. Validate full lane shape, scalar dtype, capacity, strides, offsets and owners. Dropping a scalar-only guard without implementing addressing is insufficient. |

Do not pass the gate by deleting Pack, replacing it with an unproved Bitcast,
editing only printed types, hiding it in a dump, disabling a pass, or choosing
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
This is a design example, not a claim of implemented support or a blanket ban
on Slice. Follow [Consumer indexing fusion](consumer-indexing-fusion.md).

**Weight Pack:** a pure checkpoint-weight transform or constant-zero Pack can
remain in a proven compile-time recipe with offline materialization. It must
not execute on each decode invocation. This allowance does not apply to an
activation merely because a later Pack lowering would be zero-copy.

## Acceptance evidence and reporting

Retain the inspected checkpoints and assert the structural gate on typed IR.
Then validate affected rules/passes, mixed/shared users, tuple/function ABI,
distribution, edit/resume compatibility, and the selected device kernels.
Check final buffers/source for unexpected materialization, copies, duplicated
producers or kernels. Measure compile time and runtime separately when claiming
an improvement.

Report checkpoint identity, compute Pack count per affected function (required:
zero), remaining compile-time packing and its provenance, numerical/device
results, neighboring scope checks, and any failed or unvalidated items.
The current compiler is not assumed to satisfy the policy merely because the
policy has been documented. Leave a failed gate explicit while investigating
the causal implementation; do not relabel a diagnostic checkpoint as accepted.
