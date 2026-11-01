# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical allocation candidates retain asynchronous readers until completion."""

from dataclasses import replace
import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.passes.tir import bind_prim_function_buffers
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from .synchronization.test_axis_group_inference import _chip_workspace_options
from .test_call_exit_synchronization import _kernel


def completion_graph(*, completion_scope="device"):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    tensor = fm.tensor_type("float32", (1, 64))
    small = fm.tensor_type("float32", (1,))
    produce = _kernel("produce", tensor, fm.MemoryEffect.READ)
    remote = _kernel("remote", tensor, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE), small)
    complete = _kernel("complete", tensor, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope(completion_scope)), small)
    for primitive in (produce, remote, complete):
        builder.prim_function(primitive)
    source = builder.var("source", tensor, id="source")
    other = builder.var("other", tensor, id="other")
    old = builder.call("tir.call", (source,), tensor, id="old", attrs={"callee": produce.name})
    observed = builder.call("tir.call", (old,), small, id="observed", attrs={"callee": remote.name})
    newer = builder.call("tir.call", (other,), tensor, id="newer", attrs={"callee": produce.name})
    seed = builder.call("tir.call", (other,), tensor, id="seed", attrs={"callee": produce.name})
    completed = builder.call("tir.call", (seed,), small, id="completed", attrs={"callee": complete.name})
    output = builder.call("tir.call", (newer,), tensor, id="output", attrs={"callee": produce.name})
    builder.function("main", (source, other), (observed, completed, output))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("scope", ("device", "chip"))
def test_pending_peer_read_lifetime_reaches_a_real_completion_epoch(scope):
    module = completion_graph(completion_scope=scope)
    options = replace(_chip_workspace_options(), barrier_bytes_budget=256)
    plan = fm.make_buffer_plan(module, options=options)
    nodes = function_nodes(module, module.function_map["main"])
    indices = {node.id: index for index,node in enumerate(nodes)}
    physical = plan.buffer_map["old"].mem_span.buffer
    expected = indices["completed"] - 1 if scope == "device" else len(nodes)
    assert physical.live_end == expected
    assert plan.workspace_bytes == 768
    events = plan_memory_synchronization(module, plan).events
    assert not any(event.before in {"newer", "seed"} and event.scope == "device" for event in events)
    assert any(event.before == "observed" and event.scope == "device" for event in events)
    bound = bind_prim_function_buffers(module, plan=plan)
    fm.verify_buffer_plan(replace(bound, metadata={"buffer_plan": plan.to_data()}))


@pytest.mark.parametrize("budget", (0, 255))
def test_completion_candidate_cannot_exceed_the_declared_growth_budget(budget):
    module = completion_graph()
    options = replace(_chip_workspace_options(), barrier_bytes_budget=budget)
    plan = fm.make_buffer_plan(module, options=options)
    assert plan.workspace_bytes <= 512 + budget
    events = plan_memory_synchronization(module, plan).events
    assert any(event.before in {"newer", "seed"} and event.scope == "device" for event in events)


def test_unavailable_completion_capacity_rejects_only_the_optional_proposal():
    module = completion_graph()
    original = _chip_workspace_options()
    options = replace(original, barrier_bytes_budget=256, memory_spaces=tuple(
        replace(space, maximum_bytes=512) if space.name == "workspace" else space
        for space in original.memory_spaces))
    plan = fm.make_buffer_plan(module, options=options)
    assert plan.workspace_bytes == 512
    assert any(event.scope == "device" and event.before in {"newer", "seed"}
               for event in plan_memory_synchronization(module, plan).events)
