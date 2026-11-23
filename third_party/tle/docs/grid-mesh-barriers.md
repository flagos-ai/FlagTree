# Physical mesh domains and grid barrier frames

A mesh view retains two domains. Physical IDs are encoded in the complete
node/device/cluster/block topology; slicing and reshaping do not renumber them.
The per-rank launch domain contains only cluster and block axes. Cluster member
masks first decode the complete physical ID, then project the cluster axes.
Consequently an outer rank selection cannot become a cluster selection.

The axis_group method describes independent contiguous CTA groups in a block
launch domain. Selected extents must divide their domain. The barrier descriptor
records the complete launch domain so each CTA derives the same group index and
local rank as its peers. The shard_id builtin returns zero for a unit dimension
and does not configure a hardware cluster for a block-only grid mesh.

Grid counters are allocated by the global-scratch allocation pass before LLVM
conversion and belong to the enclosing function's call frame. Calls forward an
unindexed frame base; ordinary scratch accesses add the physical CTA stride
locally, while collectives share their group's counter. This keeps counters
disjoint across static calls and preserves per-CTA scratch indexing. Native
hooks are compiled only with the __TLE__ macro enabled.

## Regression and reproducer

The old implementation of DistributedBarrierOpConversion::lowerGridBarrier in
third_party/tle/dialect/lib/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.cpp
skipped the remainder when a domain extent was one. For domain (2, 1, 2) and
groups along axis 0, the middle coordinate incorrectly retained linear_id / 2.
CTAs 0 and 2 therefore polled different counters and never completed. The
lowering must compute the remainder for every domain extent, including one.
This preserves the group implementation and its synchronization protocol.

The reproducer below hangs before that correction and completes afterward:

~~~bash
pytest python/test/tle/unit/test_tle_grid_axis_groups.py -q -k shape2
~~~

The IR regression test_tle_grid_axis_unit_domain.mlir independently requires the
unit-coordinate remainder, providing a deterministic compile-time failure before
the correction. GPU tests verify row/column groups, partial groups, helper calls,
multiple launches and 16 epochs. The grid_scratch_call_frames and grid_scratch_base
IR tests check allocation, offsets, mixed ordinary scratch and nested calls.
Existing distributed integration tests cover cluster/submesh barriers, repeated
whole-grid barriers and zero-initialized scratch.
