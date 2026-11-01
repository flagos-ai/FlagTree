# Ordered BF16 MMA accumulator

`packed_k_major_mma_accumulator` accepts scalar BF16 activations and a
K-major `(8, 2, 8)` BF16 weight packet. Its semantic accumulator is FP32,
including vector packets inferred from `ntt.packed_matmul`. The result stays
FP32 unless a typed PostOps body contains the explicit final BF16 cast. In
that case, the kernel rounds once at the store after the complete reduction.
Shared/exported FP32 results and casts across Boxing remain separate. The
optional `fuse_packed_accumulator_casts` pass uses the existing PreOps/PostOps
rewrite rules and backend capability check; automatic fusion defaults are
unchanged.

A private `Cast(ShardedView(PackedMatMul))` also fuses: the conversion executes
at the accumulator producer's ownership, then the typed result retains the
original ownership view. Both intermediate values must have a single user and
must not be exported. This does not cross Boxing, partial reductions, exclusive
ownership transitions, or shared uses. The composite rewrite avoids an inverse
cast/view commuting loop and preserves the same FP32 accumulation and BF16
rounding.

The implementation is optional, registered by
`packed_mma_accumulator_implementations()`. Existing BF16-output variants and
default preferences are unchanged. Each owner must hold the complete logical
K reduction. M and N follow the ordinary local-shard ABI; multiple local rows
and masked N/K tails are supported for one accumulation partition.

`accumulation_splits` is a positive power of two. Multiple partitions require
K to be divisible by `accumulation_splits * block_k`. Every partition starts
with a zero FP32 MMA2 accumulator, consumes ascending K tiles, selects the
logical row, and merges that partial with `libdevice.add_rn` in ascending
partition order. Partitions are sequential within one CTA, with no partial
buffer or cross-CTA reduction. They describe an accumulation algorithm rather
than weight quantization blocks or distribution axes.

The native MLA experiment uses K128 and two partitions for Q_A, KV_A and each
TP-local o_proj, and one for Q_B. Those choices are explicit local candidates
selected before lowering. The generic implementation contains no model IDs.

`weight_load_layout="coalesced"` gives global weight loads a blocked layout
matching the N8/K16 packet: contiguous K pairs span eight lanes, four N rows
span the remaining lanes, and independent N/K regions span the warps. The
conversion to the MMA operand then replicates the values required by that
fragment. Propagating the MMA layout directly to global pointers repeats
addresses in unused warp groups for an N16 tile and extends pointer live
ranges. The explicit load layout retains the asynchronous pipeline and K128
arithmetic. `"mma"` preserves the old direct layout for existing artifacts
whose parameters omit this choice and provides a controlled comparison.

Regression coverage is in `test_mma_accumulator.py`: normal compiler selection,
edit/resume, ordinary artifact rendering, invalid partitions, 128-CTA device
execution, multiple rows, tails, output guards and FP32 precision. The offline
weight experiment also needs its own full-model token/state and performance
acceptance; these kernel tests do not establish an end-to-end gain.
`test_accumulator_cast_fusion.py` covers the explicit cast through the normal
pass, selection, artifact and device paths, along with shared/exported values,
Boxing, partial projections and unsupported PostOps.
`test_accumulator_cast_view_fusion.py` adds ownership refinement/coarsening,
non-first chip axes, shared/exported intermediates, edit/resume, independent
numerical evaluation and reusable device execution with full output guards.
