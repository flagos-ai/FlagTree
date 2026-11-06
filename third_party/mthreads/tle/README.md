# SQMMA shared-memory view compatibility

MThreads TLE SQMMA consumers can share an allocation only when they infer
exactly the same root `MemDescType`. This includes the shape, element type,
shared-memory encoding, memory space, mutability, and allocation shape.
Different encodings are not accepted merely because their address mappings
might be equivalent. Logical operand direction belongs to the final view;
it does not impose a separate row/column contract on the allocation root.

For example, FP8 Q `[32, 256]`, KV `[64, 256]`, and P `[32, 64]` support
`wgmma(Q, KV, trans_b=True)` and `wgmma(P, KV)` using one physical KV allocation.
The corresponding BF16 configuration requires incompatible physical swizzles
and is rejected. This is a restriction on those inferred layouts, not a ban
on BF16 sharing: two untransposed BF16 consumers of a `[64, 64]` tile can share
the same root. This positive case does not establish BF16 K-transpose/V reuse.

A staged allocation retains its complete stage shape even when a consumer
uses a subslice. Captures of either the allocation or a stage view must match
the corresponding block arguments in every warp-specialize partition,
including partitions that do not use the capture. A shared physical layout
does not remove synchronization requirements: producers must complete writes
before either consumer reads, and both SQMMA consumers must complete before
the producer overwrites that stage. `tle.pipe` provides the stage publication
and reuse protocol in the dynamic pipeline regression.

The compile and numerical regressions are in
`third_party/mthreads/python/test/unit/tle/test_tle_sqmma_kv_root.py`.
