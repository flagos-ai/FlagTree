# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Kernel-private temporaries still create hazards in caller-owned arenas."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import MemorySharingScope
from triton.flagmega.passes.tir import materialize_kernel_prim_functions
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from triton.flagmega.passes.tir.bufferize.call_accesses import CallAccessResolver
from triton.flagmega.passes.tir.bufferize.synchronization import _node_accesses
from .test_kernel_workspace_abi import _selected_workspace_kernels
from .test_memory_synchronization import _workspace_options


@pytest.mark.parametrize("scope,expected", [(MemorySharingScope.CHIP, "grid"), (MemorySharingScope.BLOCK, "block")])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_reused_workspace_waits_for_all_previous_readers(scope, expected, level):
    module = _selected_workspace_kernels()
    plan = fm.make_buffer_plan(module, options=replace(_workspace_options(scope), optimization_level=level))
    assert plan.optimization_level == level
    first = plan.resolve_kernel_workspace("first", "partials")
    second = plan.resolve_kernel_workspace("second", "partials")
    assert first.offset == second.offset
    assert first.live_end < second.live_start
    events = plan_memory_synchronization(module, plan).events
    assert len(events) == 1
    event, = events
    assert (event.after, event.before, event.scope) == ("first", "second", expected)
    assert event.ranges[0].offset == first.offset
    assert event.ranges[0].nbytes == first.nbytes


def workspace_then_output(*, retained=False):
    builder = fm.IRBuilder(dialect="semantic_tir", stage="selected_tir")
    tensor = fm.tensor_type("float32", (32, ))
    source = builder.var("source", tensor, id="source")
    attrs = {
        "semantic_op": "math.silu", "candidate": "tir.unary.local", "parameters":
        {"family": "unary", "variant": "local"}, "facts": {}, "semantic_attrs": {}
    }
    scratch_attrs = {
        **attrs, "parameters": {
            **attrs["parameters"], "workspaces": ({
                "name": "scratch",
                "type": tensor,
                "memory_space": "workspace",
                "alignment": 64,
                "lifetime": "function" if retained else "invocation",
            }, )
        }
    }
    first = builder.call("tir.kernel", (source, ), tensor, id="first", attrs=scratch_attrs)
    reused = builder.call("tir.kernel", (source, ), tensor, id="reused", attrs=attrs)
    result = builder.call("tir.kernel", (reused, ), tensor, id="result", attrs=attrs)
    builder.function("main", (source, ), (first, result))
    return materialize_kernel_prim_functions(builder.build(entry="main"))


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_workspace_to_output_reuse_is_synchronized_only_when_ranges_overlap(retained, level):
    module = workspace_then_output(retained=retained)
    options = replace(_workspace_options(MemorySharingScope.CHIP), optimization_level=level)
    plan = fm.make_buffer_plan(module, options=options)
    scratch = plan.resolve_kernel_workspace("first", "scratch")
    bindings = dict(plan.function_map["main"].values)
    output = plan.buffer_map[bindings["reused"][0]]
    overlaps = scratch.offset < output.offset + output.nbytes and output.offset < scratch.offset + scratch.nbytes
    assert overlaps is not retained
    events = [event for event in plan_memory_synchronization(module, plan).events if event.before == "reused"]
    if retained:
        assert not events
    else:
        assert len(events) == 1
        assert events[0].scope == "grid"
        assert events[0].after == "first"


def test_workspace_access_is_a_concrete_read_write_range():
    module = _selected_workspace_kernels()
    plan = fm.make_buffer_plan(module, options=_workspace_options(MemorySharingScope.CHIP))
    scratch = plan.resolve_kernel_workspace("first", "partials")
    accesses = _node_accesses(module, plan, "main", module.node_map["first"], dict(plan.function_map["main"].values))
    access, = (access for access in accesses if access.buffer == scratch.id)
    assert access.mode == "read_write"
    assert access.physical_id == "workspace:@main"
    assert (access.offset, access.nbytes) == (scratch.offset, scratch.nbytes)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_callee_scratch_is_translated_into_the_callers_arena(nested, level):
    base = workspace_then_output()
    worker = replace(base.functions[0], name="worker")
    tensor = base.node_map["source"].type
    pair_type = fm.TupleType((tensor, tensor))
    nodes = list(base.nodes)
    functions = [worker]
    callee = "worker"
    if nested:
        outer = fm.Node("outer", "builtin.var", (), tensor)
        forwarded = fm.Node("forwarded", "tir.call", (outer.id, ), pair_type, attrs={"callee": callee})
        first = fm.Node("forwarded.first", "builtin.get_item", (forwarded.id, ), tensor, attrs={"index": 0})
        second = fm.Node("forwarded.second", "builtin.get_item", (forwarded.id, ), tensor, attrs={"index": 1})
        nodes.extend((outer, forwarded, first, second))
        functions.append(fm.Function("wrapper", (outer.id, ), (first.id, second.id)))
        callee = "wrapper"
    entry = fm.Node("entry_source", "builtin.var", (), tensor)
    invocation = fm.Node("invocation", "tir.call", (entry.id, ), pair_type, attrs={"callee": callee})
    nodes.extend((entry, invocation))
    functions.append(fm.Function("main", (entry.id, ), (invocation.id, )))
    module = replace(base, nodes=tuple(nodes), functions=tuple(functions))
    options = replace(_workspace_options(MemorySharingScope.CHIP), optimization_level=level)
    plan = fm.make_buffer_plan(module, options=options)
    scratch = plan.resolve_kernel_workspace("first", "scratch")
    accesses = CallAccessResolver(module, plan, _node_accesses).accesses("main", invocation,
                                                                         dict(plan.function_map["main"].values))
    access, = (access for access in accesses if access.buffer == scratch.id)
    frame_offset = plan.call_map["invocation"].memory_pool_map["workspace"].offset
    if nested:
        frame_offset += plan.call_map["forwarded"].memory_pool_map["workspace"].offset
    assert access.node == invocation.id
    assert access.physical_id == "workspace:@main"
    assert access.mode == "read_write"
    assert (access.offset, access.nbytes) == (frame_offset + scratch.offset, scratch.nbytes)
