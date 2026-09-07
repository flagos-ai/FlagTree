# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Memory placement may remove reuse-only barriers without raising any pool peak."""

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import (
    AllocationStrategy, BufferPlan, MemoryAllocationScope, MemorySharingScope, MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import BufferPlanner, BufferizationOptions, plan_buffers, plan_memory_synchronization


def _case(scope=MemorySharingScope.CHIP, intrinsic=False):
    options = BufferizationOptions((
        MemorySpace("workspace", "device", 64, 1 << 20, AllocationStrategy.SAT,
                    allocation_scope=MemoryAllocationScope.FUNCTION, sharing_scope=scope),
        MemorySpace("rdata", "readonly_device", 64, 1 << 20, AllocationStrategy.LINEAR,
                    allocation_scope=MemoryAllocationScope.MODULE, sharing_scope=MemorySharingScope.CHIP),
        MemorySpace("external", "external", 1, 1 << 20, AllocationStrategy.EXTERNAL,
                    allocation_scope=MemoryAllocationScope.EXTERNAL, sharing_scope=MemorySharingScope.CHIP),
    ))
    placement = fm.Placement((2, 4), "yx", "bb")
    typ = fm.tensor_type("float32", (16,))
    distributed = fm.DistributedType(typ, (fm.SBP.broadcast(),), placement, fm.SBP.partial((0, 1)))
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", distributed, id="source")
    other = builder.var("other", typ, id="other")
    # This earlier non-overlapping working set fixes a 4096-byte minimum peak.
    anchor = builder.call("tir.kernel", (other,), fm.tensor_type("float32", (1024,)), id="anchor")
    anchor_done = builder.call("tir.kernel", (anchor,), typ, id="anchor_done")
    produced = builder.call("tir.kernel", (source,), distributed, id="produced")
    consumed = builder.call("tir.kernel", (produced,), typ, id="consumed")
    late = builder.call("tir.kernel", (consumed,) if intrinsic else (other,), typ, id="late",
                        attrs={"semantic_op": "distributed.boxing", "facts": {"collective_semantics": "all_gather"}}
                        if intrinsic else {})
    output = builder.call("tir.kernel", (late,), typ, id="output")
    builder.function("main", (source, other), (anchor_done, consumed, output))
    return builder.build(entry="main"), options


def _grids(module, plan):
    return tuple(value for value in plan_memory_synchronization(module, plan).events if value.scope == "grid")


def test_planner_uses_same_peak_to_remove_reuse_only_grid_barrier():
    module, options = _case()
    baseline = BufferPlanner(module, options).run()
    assert any(value.before == "late" for value in _grids(module, baseline))
    selected = plan_buffers(module, options=options)
    assert selected.workspace_bytes == baseline.workspace_bytes
    assert len(_grids(module, selected)) < len(_grids(module, baseline))
    assert not any(value.before == "late" for value in _grids(module, selected))
    assert selected.allocator == baseline.allocator
    assert BufferPlan.from_data(selected.to_data()) == selected


def test_block_local_reuse_has_no_chip_barrier_to_avoid():
    module, options = _case(MemorySharingScope.BLOCK)
    baseline = BufferPlanner(module, options).run()
    selected = plan_buffers(module, options=options)
    assert selected == baseline


def test_intrinsic_collective_dependency_must_remain_synchronized():
    module, options = _case(intrinsic=True)
    selected = plan_buffers(module, options=options)
    assert any(value.before == "late" for value in _grids(module, selected))
