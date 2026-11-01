# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Memory placement may remove reuse-only barriers without raising any pool peak."""

from triton.flagmega import ir as fm
import ast
from collections import Counter
from dataclasses import replace

import pytest
from triton.flagmega.ir.bufferization import (
    AllocationStrategy,
    BufferPlan,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import (
    BufferizationOptions,
    BufferPlanner,
    plan_buffers,
    plan_memory_synchronization,
)


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


def test_a_required_collective_cut_does_not_add_an_unhelpful_reuse_preference():
    from triton.flagmega.passes.tir.bufferize.reuse_preferences import collect_reuse_preferences

    module, options = _case(intrinsic=True)
    baseline = BufferPlanner(module, options).run()
    assert any(value.before == "late" for value in _grids(module, baseline))
    produced = baseline.buffer_map["produced"].mem_span.buffer.id
    late = baseline.buffer_map["late"].mem_span.buffer.id
    preferences = collect_reuse_preferences(module, baseline).get(("main", "workspace"), ())
    # Relocating these allocations cannot remove the collective's true data
    # dependency. Pricing that pair sends SAT after a barrier it cannot remove.
    assert tuple(sorted((produced, late))) not in preferences


def _emitted_barriers(scopes):
    """Independently count the executor's actual device/chip/CTA operations."""
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

    body = ['{% import "entrypoints/_common.py.jinja" as entry with context %}']
    for scope in scopes:
        body.append({"device": "{{ entry.device_barrier() }}", "grid": "{{ entry.stage_barrier() }}",
                     "block": "    tl.debug_barrier()"}[scope])
    rendered = TritonTemplateRegistry().environment.from_string("\n".join(body)).render(
        distributed_entry=True, mesh_requires_device_dptr=True)
    tree = ast.parse("def witness():\n" + rendered + "\n")
    counts = Counter()
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr == "debug_barrier":
            counts["block"] += 1
        elif call.func.attr == "distributed_barrier":
            device = any(k.arg == "space" and isinstance(k.value, ast.Constant) and k.value.value == "device"
                         for k in call.keywords)
            counts["device" if device else "grid"] += 1
    return counts


@pytest.mark.parametrize("before_scopes,after_scopes,accepted", [
    (("device",), ("grid", "block"), True),
    # The captured MLA regression: three fewer remote epochs, one more CTA fence.
    (("device",) * 3, ("grid",) * 3 + ("block",), True),
    (("device",), ("grid",) * 3 + ("block",), False),
    (("grid",) * 4, ("device",), False),
    (("block",), ("grid",), False),
])
def test_memory_dominance_uses_emitted_rendezvous(monkeypatch, before_scopes, after_scopes, accepted):
    from triton.flagmega.ir.bufferization import MemorySynchronizationPlan, SynchronizationEvent
    from triton.flagmega.passes.tir.bufferize import synchronization
    from triton.flagmega.passes.tir.bufferize.reuse_preferences import dominates_memory_schedule

    module, options = _case()
    baseline = BufferPlanner(module, options).run()
    candidate = replace(baseline)

    def sequence(scopes):
        return MemorySynchronizationPlan(tuple(SynchronizationEvent(
            "main", "produced", "consumed" if i == 0 else "output", scope, (), (),
        ) for i, scope in enumerate(scopes)))

    # Isolate the selector from hazard discovery. Its input is the typed
    # synchronization plan; the independent lowering above supplies the cost
    # witness rather than repeating the selector's numerical weights.
    monkeypatch.setattr(synchronization, "plan_memory_synchronization",
                        lambda mod, plan: sequence(after_scopes if plan is candidate else before_scopes))
    if accepted:
        before = _emitted_barriers(before_scopes)
        after = _emitted_barriers(after_scopes)
        assert after["device"] < before["device"]
        assert sum(after.values()) < sum(before.values())
    assert dominates_memory_schedule(module, candidate, baseline) is accepted
