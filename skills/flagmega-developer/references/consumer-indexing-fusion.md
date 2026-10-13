# Consumer indexing fusion

Use this reference when pure indexing materializes a private consumer input,
or related state updates separately load partitions of one source. Examples
include Slice/Gather into cache writes, Q/K/V or gate/value deinterleaving,
cropping into a downstream kernel, and layout adapters around publication.
These are design methods; examples do not establish installed capabilities or
authorize implementation beyond the user's request.

## Background and expected advantages

A coarse producer can still leave an expensive boundary: it emits a combined
or packed tensor, while consumers request separate dense fields. A Slice is
semantically necessary selection, but its value need not be stored as a new
tensor. The missing contract is often how a consumer reads its source, rather
than how the producer computes it.

Inspect actual lowering. A reshape-like graph, direct consumer edge, pure
evaluator, or test named "without a copy" does not prove a storage view.
Selected copy kernels, allocations and scalar addresses are the evidence.

Composing indexing with consumer loads can remove intermediate reads/writes,
temporary results and device calls. Grouping related writes may also remove
duplicate metadata work and express ownership more clearly to distribution
and scheduling. Smaller graphs can improve compiler iteration, but this needs
measurement. Longer source lifetimes, repeated gathers, registers, code size,
communication or synchronization can offset the saving. An IR/TIR call is not
necessarily a separate GPU launch; distinguish those counts.

## Choose the abstraction by the missing contract

| Situation | Design |
| --- | --- |
| Consumer needs a producer-owned output representation | Extend producer output support and demand propagation; follow [Representation optimization](representation-optimization.md). |
| Private pure indexing only changes what an existing consumer reads | Extend its input access contract or supported input fusion. Keep the selection inside consumer addressing. |
| Several writes share a source, destination state and update epoch | Consider one semantic operation owning the coordinated update, including metadata and completion phases. |
| Selection is shared/exported or needs an independently reusable value | Consider a first-class strided view with a complete alias/storage ABI, or retain necessary materialization. |

Do not create a new operation merely to hide Slice. Reuse an existing contract
when it expresses the behavior; define a new one when coordination or source
interpretation is a real semantic unit. PreOps attributes are useful only if
the selected family actually implements indexing fusion. A blanket ban on all
Slice/Gather is not this method's acceptance rule.

## Prove the read mapping and legal region

For each consumer coordinate `y`, derive the semantic source coordinate `f(y)`;
then map `f(y)` through the **source buffer's** physical ABI. Separately derive
the destination's logical coordinate, ownership and physical offset. Record
axis roles, strides, starts/steps, bounds, scalar dtype and complete lane shape.
Include masks for padded capacity and dynamic active regions.
Keep runtime index operands and their read/lifetime dependencies explicit;
preserve the original bounds, negative-index and masked-access semantics.

Keep scalar units explicit. With a contiguous final-axis lane product `L`,
scalar coordinate `i` maps to group `i // L` and lane `i % L`. Other mappings
need their own proof. A partition boundary inside a vector packet requires
correct lane selection/masking or an explicit capability rejection.

A per-head field slice changes its extent but retains the source's head
stride. A base-pointer offset alone cannot turn interleaved fields into a
dense tensor. For an alias view, declare byte offset, physical strides, backing
MemSpan and lifetime. Holes between rows mean dense element count is not the
addressed byte span. Public results and consumers with fixed layout contracts
must receive compatible storage.

For a structural fusion, prove the field ranges, order and coverage; compatible
layouts, state/layer/epoch operands; absence of intermediate state observers;
and legality of moving the reads/writes. Preserve shared/exported users without
duplicating expensive producers. Numeric conversions, RoPE or normalization
between selection and consumption need explicit numerical support; they are
not pure indexing. Unproved cases limit applicability, rather than permitting
silent precision changes or an unsupported fused kernel.

## Put formation, distribution and execution at their owning stages

Import a natural semantic unit directly, and provide structural rules for
compatible older graphs where useful. Form coordinated regions before packing
or AutoDistribution when that exposes their legal axes and costs. Recognize
patterns by contracts, not model names or SSA prefixes. Lower their proven
access maps through the ordinary physical ABI afterward.

Allow distributions supported by the actual computation. A whole-head or
partition ownership can be hidden by a flat indexing expansion; model the
semantic axes so the solver can see it. Preserve required partial reductions,
Boxing and publication. A local consumer read map cannot supply missing remote
data or make unfinished partial values safe for state writes. Producer rounding
and collective order remain part of the contract even when a buffer vanishes.

For coordinated state updates:

- Read the shared old position/version consistently and declare state effects,
  allocation scope and argument/field partitions in IR and TIR.
- Elect unique writers for each destination at its actual block/chip scope;
  map local indices to logical owners through each buffer's ABI.
- Merge metadata writes only when their intermediate observations are absent
  and the final observable state is equivalent.
- Complete all required payload writes before advancing a shared cursor or
  publishing completion. Keep synchronization required by later readers,
  including when no cursor advance is requested. One op may still need several
  execution phases.

Expose full source access and lifetime to the normal memory planner. Retain
typed packed inputs through subsequent rewrites; do not reconstruct a scalar
Slice/Pack chain during vector-contract lowering. Complete the API, evaluator,
serialization, candidates, codegen and verification through the
[IR extension SOP](ir-extension-sop.md).

## Implemented example: combined KV to paired cache writes

Assume a source already rounded to BF16, shaped `[T,H,2D]`, with each head's
payload ordered `[K(D) | V(D)]`. The current cache ABI has equal K/V head
dimensions. The semantic chain is:

```text
kv = Projection -> required Boxing -> head view
k = Slice(kv, K); v = Slice(kv, V)
s1 = KeyUpdate(k, s0, advance=false)
s2 = ValueUpdate(v, s1, advance=requested)
```

`F.nn.update_paged_attention_kv_cache_pair(kv, state, layer, advance, layout=...)`
replaces the two selections and updates with one coordinated region:

```text
Projection -> required Boxing -> head view -> paired cache update
```

Derive `H,D` and cache packing from the typed state. Validate the combined input
geometry and supported scalar/vector representations. Read K at scalar `d` and
V at `D+d` within the same source head, using its original physical stride;
write the corresponding cache fields directly. Advance the sequence once,
after both writes and required synchronization. The input packing width may
differ from the cache's; implement the scalar coordinate mapping explicitly.
Unequal K/V dimensions require an appropriate state contract rather than an
extra attribute pretending the current cache ABI supports them.

Formation requires complementary unit-step partitions from the same source,
compatible layer/layout, a non-advancing first update and no observer of the
intermediate state. Leave the expensive projection and necessary reduction
boundaries intact. Core ownership is in
`ir/ops/nn/update_paged_attention_kv_cache_pair.py`; the older-IR formation rule
is `rules/neutral/form_paged_attention_kv_cache_pair.py`, registered in
`passes/target_independent.py` before AutoPacking. The importer emits the pair
directly. Native vector retention, distribution candidate inference, semantic
TIR selection, the portable implementation catalog and source-coordinate
renderer preserve the operation through the ordinary compiler pipeline.
All chip replicas require a complete, non-partial source; legal block splits
remain available. Shared K/V selections survive for their other consumers.
The selected write kernel uses one old epoch for both fields and one internal
grid barrier before an optional sequence advance. Verify those contracts in
the actual emitted implementation when extending it.

Regression coverage includes scalar and multi-component vector inputs, all
six axis layouts, differing input/cache lane counts, complete cache comparison,
cross-page chunks, block-cyclic tails, reusable calls, and chip replicas.
Structural negative cases preserve intermediate-state observers and required
partial Boxing. The full GLM checkpoint must retain zero compute Pack and zero
MLA Slice at packing, distribution and pre-TIR boundaries; focused kernel tests
alone do not certify complete model execution.

For `B = T * H * 2D * sizeof(BF16)`, eliminating the two Slice copies removes
`B` bytes of logical temporary results and `2B` bytes of copy traffic. At
`T=1,H=64,D=256`, these are 64 KiB and 128 KiB respectively. These figures are
logical accounting, not measured peak-memory or latency savings. Physical
replicas, buffer reuse and the retained source lifetime affect actual costs.

## Acceptance evidence

Retain a minimal materialization reproducer and a structural fail-before test,
then inspect complete IR/TIR, generated source and buffers after the change.
For the targeted private chain, check that copy calls and temporary results
are gone, the shared producer remains single, and complete packed addressing
survives lowering. Apply [IR acceptance](ir-acceptance.md), including zero
compute Pack; do not impose zero Slice on unrelated necessary transforms.

For state writes, compare independently initialized reference/candidate storage
with nonzero data and untouched-region sentinels. Cover repeated updates,
advance on/off, chunks and page boundaries, nonidentity page tables, multiple
owners, tails, scalar/packed lanes, aliases and reusable calls. Include rejected
regions with shared intermediate state or incompatible ranges/contracts.
Measure the full consumer region and requested workload, including retained
communication and synchronization; follow [Performance iteration](performance-iteration.md).
