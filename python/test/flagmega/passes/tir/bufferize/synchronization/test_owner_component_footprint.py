# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Chip-wide alias hazards must cover every compact owner component."""

from dataclasses import replace
from math import prod

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import (
    AllocationStrategy, MemSpan, MemoryAllocationScope, MemorySharingScope, MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions, plan_memory_synchronization
from triton.flagmega.passes.tir.bufferize.synchronization import _node_accesses


def _options(scope):
    return BufferizationOptions((
        MemorySpace("workspace", "device", 64, 1 << 20, AllocationStrategy.SAT,
                    allocation_scope=MemoryAllocationScope.FUNCTION, sharing_scope=scope),
        MemorySpace("rdata", "readonly_device", 64, 1 << 20, AllocationStrategy.LINEAR,
                    allocation_scope=MemoryAllocationScope.MODULE, sharing_scope=MemorySharingScope.CHIP),
        MemorySpace("external", "external", 1, 1 << 20, AllocationStrategy.EXTERNAL,
                    allocation_scope=MemoryAllocationScope.EXTERNAL, sharing_scope=MemorySharingScope.CHIP),
    ))


def _module(partial, mesh=(2, 4)):
    placement = fm.Placement(mesh, "yx", "bb")
    tensor = fm.tensor_type("float32", (16 if partial else 16 * prod(mesh),))
    distributed = fm.DistributedType(
        tensor, (fm.SBP.broadcast() if partial else fm.SBP.split_contiguous((0, 1)),),
        placement, fm.SBP.partial((0, 1)) if partial else None,
    )
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    source = builder.var("source", distributed, id="source")
    other = builder.var("other", fm.tensor_type("float32", (16,)), id="other")
    produced = builder.call("tir.kernel", (source,), distributed, id="produced")
    consumed = builder.call("tir.kernel", (produced,), tensor, id="consumed")
    late = builder.call("tir.kernel", (other,), other.type, id="late")
    output = builder.call("tir.kernel", (late,), other.type, id="output")
    builder.function("main", (source, other), (consumed, output))
    return builder.build(entry="main")


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize("scope", (MemorySharingScope.CHIP, MemorySharingScope.BLOCK))
@pytest.mark.parametrize("mesh", ((2, 4), (3, 5)))
def test_access_range_covers_the_physical_owner_domain(partial, scope, mesh):
    module = _module(partial, mesh)
    plan = fm.make_buffer_plan(module, options=_options(scope))
    bindings = dict(plan.function_map["main"].values)
    descriptor = plan.buffer_map[bindings["produced"][0]]
    access = next(value for value in _node_accesses(module, plan, "main", module.node_map["produced"], bindings)
                  if value.buffer == descriptor.id)
    assert descriptor.nbytes == 64
    assert access.nbytes == (64 * prod(mesh) if scope is MemorySharingScope.CHIP else 64)


def test_reuse_inside_nonzero_owner_requires_grid_war_barrier():
    module = _module(True)
    plan = fm.make_buffer_plan(module, options=_options(MemorySharingScope.CHIP))
    bindings = dict(plan.function_map["main"].values)
    produced = plan.buffer_map[bindings["produced"][0]]
    late = plan.buffer_map[bindings["late"][0]]
    assert produced.mem_span.buffer.live_end < late.mem_span.buffer.live_start
    # A legal lifetime reuse of only owner one's range. Owner zero's MemSpan
    # is disjoint, but the all-owner read can race with this new allocation.
    moved = replace(late.mem_span.buffer, start=produced.offset + 64)
    assert moved.offset + moved.nbytes <= produced.mem_span.buffer.offset + produced.mem_span.buffer.nbytes
    plan = replace(plan, physical_buffers=tuple(moved if value.id == moved.id else value for value in plan.physical_buffers),
                   buffers=tuple(replace(value, mem_span=MemSpan(moved, value.mem_span.start, value.mem_span.size))
                                 if value.physical_id == moved.id else value for value in plan.buffers))
    events = plan_memory_synchronization(module, plan).events
    event = next((value for value in events if value.before == "late"), None)
    assert event is not None, "Owner-zero-only analysis misses the reused owner-one range"
    assert event.scope == "grid"
    assert "READ->WRITE" in event.hazards
    assert any(value.offset == produced.offset + 64 and value.nbytes == 64 for value in event.ranges)
