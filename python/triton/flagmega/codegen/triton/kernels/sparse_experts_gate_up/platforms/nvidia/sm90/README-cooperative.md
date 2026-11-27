# Ordered FP8 scale-block cooperation

`cooperative_wgmma` implements the existing `nn.sparse_experts_gate_up`
operation. It is optional: the generic SM90 catalog does not register it.
The workload profile chooses it through an ordinary selection plan, before
LowerTIR, allocation and synchronization planning.

## Numerical contract

The original WGMMA implementation serializes all K128 scale blocks inside
each route/row owner. Each raw dot is independent, but its scale merge is
ordered. The cooperative implementation distributes these raw blocks across
the chip and preserves the merge:

1. Quantize the same BF16 K128 activation block with UE8M0, use the same FP8
   WGMMA geometry (64 weight rows and 16 identical activation columns), then
   multiply each FP32 raw result by its original B scale with FP32 rounding.
2. Publish raw results and activation scales through explicitly typed scratch
   and one full chip barrier.
3. Each original output owner consumes real K blocks in their original order,
   using one `fma(raw_times_B, A_scale, accumulator)` per block. The padded
   register extent adds no arithmetic steps. Apply the original input scales,
   projection conversion, clamps, SiLU and activation conversion.

The algorithm does not replace the FP8 dot with a SIMT reduction, reassociate
the K-block accumulation, combine A/B scaling, or remove BF16 roundings.

## Representation, distribution and effects

The implementation requires fixed extents, complete K divisible by 128,
BF16 activation, FP8 E4M3 weights and FP32 128x128 block scales, as required
by the original WGMMA candidate. Route and row ownership use one contiguous
split stage; scales are broadcast. Packed BF16 activation/results retain their
scalar lane maps. Partial/exclusive values and unproved physical views are
rejected. Existing IR legality, including divisible contiguous distribution
dimensions, still applies.

The task domain enumerates the chip's route and row owners, preserving all
non-block placement coordinates. The renderer uses the real per-owner
coordinate maps and compact component stride; it never interprets compact
weights as canonical global storage. All peer inputs, outputs and phase
scratch must be physically chip shared, without a per-block pool stride,
unknown parent backing or runtime view offset. A peer read never selects a
different chip. The candidate declares `source_memory_scope=chip`, allowing
normal storage and hazard analysis to see its complete read participation.

Two invocation-lifetime FP32 workspaces are declared before selection:

- `projection_partials[T, chip_routes, 2, ceil(chip_rows/64), K/128, 64]`;
- `activation_scales[T, chip_routes, K/128]`.

Both have canonical contiguous chip-shared storage. Each raw element has one
writer. Only the first row tile publishes activation scales. Scratch reuse
after the merge remains subject to normal caller hazard analysis. Separate
noinline partial/merge functions keep register live ranges bounded; their
wrapper owns the entry-relative grid barrier. Every CTA reaches that barrier.
Readonly weights, offline packing recipes and the external model ABI stay
unchanged; new scratch still requires rebuilding downstream memory/sync plans.

## Validation

`python/test/flagmega/codegen/triton/kernels/sparse_experts/test_cooperative_wgmma.py`
uses ordinary compilation, serialization, artifact publication and runtime
loading. It checks explicit workspace/effects, unsupported quantization,
generic-target isolation, C0/C3 compact component/scale addresses, negative
physical-sharing cases, full bitwise outputs against the original WGMMA,
multiple input/route/scale updates, packed8, 16 warps, K1536's 12-block merge,
owner N tails, a sharded successor, guards and zero spills. Neighbor tests
exercise the original gate/up and partitioned-down implementations.

The initial valid selected-TIR reproducer failed in ordinary publication
because no `cooperative_wgmma` template existed; the normal renderer/template
path supplies that support. Invalid fixture/import/type errors are recorded
separately from compiler or performance failures. Local stage gains do not
establish full-model correctness or speed: the complete call graph must pass
resource checks, independent official greedy requests and state/cache checks,
followed by a same-card runtime comparison.
