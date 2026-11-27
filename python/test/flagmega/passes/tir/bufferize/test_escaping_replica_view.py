# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Publication changes a private replica and all its views to caller storage."""

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import AllocationStrategy, MemoryAllocationScope, MemorySharingScope, MemorySpace
from triton.flagmega.passes.tir.bufferize import BufferizationOptions


def test_escaping_shard_view_recomputes_backing_after_result_promotion():
    options = BufferizationOptions((
        MemorySpace("workspace", "device", 64, 1 << 20, AllocationStrategy.SAT,
                    allocation_scope=MemoryAllocationScope.FUNCTION, sharing_scope=MemorySharingScope.BLOCK),
        MemorySpace("rdata", "readonly_device", 64, 1 << 20, AllocationStrategy.LINEAR,
                    allocation_scope=MemoryAllocationScope.MODULE, sharing_scope=MemorySharingScope.CHIP),
        MemorySpace("external", "external", 1, 1 << 20, AllocationStrategy.EXTERNAL,
                    allocation_scope=MemoryAllocationScope.EXTERNAL, sharing_scope=MemorySharingScope.CHIP),
    ))
    placement = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("bfloat16", (3, 128))
    broadcast = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    split = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), placement)
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value = builder.var("value", broadcast, id="value")
    produced = builder.call("math.add", (value, value), broadcast, id="produced")
    view = builder.call("distributed.sharded_view", (produced,), split, id="view", attrs={"new_type": split})
    builder.function("main", (value,), (view,))
    plan = fm.make_buffer_plan(builder.build(entry="main"), options=options)
    source, result = (plan.buffer_map[name] for name in ("produced", "view"))
    assert source.mem_span.must_alias(result.mem_span)
    assert source.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert result.distributed_storage_kind is fm.DistributedBufferStorageKind.CANONICAL_GLOBAL
    assert source.nbytes == result.nbytes == 3 * 128 * 2
    assert source.storage == result.storage == "output"
