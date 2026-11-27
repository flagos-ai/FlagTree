# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A callee's real completion fences must survive its effect summary."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from .synchronization.test_axis_group_inference import _chip_workspace_options


def _kernel(name, value_type, input_effect, result_type=None):
    result_type = value_type if result_type is None else result_type
    reads = ("input",) if input_effect.physical_mode & fm.MemoryAccessMode.READ else ()
    writes = ("input", "result") if input_effect.physical_mode & fm.MemoryAccessMode.WRITE else ("result",)
    role = (fm.PrimParameterRole.INOUT if input_effect.physical_mode & fm.MemoryAccessMode.WRITE
            else fm.PrimParameterRole.INPUT)
    dispatch = fm.KernelDispatch(
        semantic_op=f"test.{name}", candidate=f"test.{name}",
        parameters={"family": "test", "variant": name},
        arguments=("input",), outputs=("result",), reads=reads, writes=writes,
        memory_effects=(("input", input_effect), ("result", fm.MemoryEffect.WRITE)),
    )
    return fm.PrimFunction(
        name, "triton",
        (fm.PrimParameter("input", value_type, role),
         fm.PrimParameter("result", result_type, fm.PrimParameterRole.OUTPUT)),
        fm.Sequential((dispatch,)),
        fm.Return((fm.ReturnBinding(fm.ValueRef("result", result_type), "result"),)),
    )


def completed_read_graph(*, completion_scope="device", tail_read=False, nested=False,
                         read_scope="device", write_scope="inferred", prior_caller_read=False,
                         second_call=False):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    placement = fm.Placement((2, 4), "cx", "cb")
    value_type = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), placement,
    )
    remote = _kernel("remote", value_type, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope(read_scope)))
    completion = _kernel("completion", value_type, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope(completion_scope)))
    writer = _kernel("writer", value_type, fm.MemoryEffect(
        fm.MemoryAccessMode.WRITE, fm.MemoryAccessScope(write_scope)))
    for kernel in (remote, completion, writer):
        builder.prim_function(kernel)
    formal = builder.var("formal", value_type, id="formal")
    read = builder.call("tir.call", (formal,), value_type, id="read",
                        attrs={"callee": "remote"})
    completed = builder.call("tir.call", (read,), value_type, id="completed",
                             attrs={"callee": "completion"})
    outputs = [completed]
    if tail_read:
        outputs.append(builder.call("tir.call", (formal,), value_type, id="tail",
                                    attrs={"callee": "remote"}))
    result_type = value_type if len(outputs) == 1 else fm.TupleType(tuple(value_type for _ in outputs))
    builder.function("worker", (formal,), outputs)
    callee = "worker"
    if nested:
        forwarded = builder.var("forwarded", value_type, id="forwarded")
        wrapped = builder.call("tir.call", (forwarded,), result_type, id="wrapped",
                               attrs={"callee": "worker"})
        builder.function("wrapper", (forwarded,), (wrapped,))
        callee = "wrapper"
    source = builder.var("source", value_type, id="source")
    extra = builder.var("extra", value_type, id="extra")
    results = []
    if prior_caller_read:
        results.append(builder.call("tir.call", (extra,), value_type, id="previous",
                                    attrs={"callee": "remote"}))
    invocation = builder.call("tir.call", (source,), result_type, id="invocation",
                              attrs={"callee": callee})
    results.append(invocation)
    if second_call:
        results.append(builder.call("tir.call", (extra,), result_type, id="invocation2",
                                    attrs={"callee": callee}))
    reused = builder.call("tir.call", (extra if prior_caller_read else source,), value_type, id="reused",
                          attrs={"callee": "writer"})
    builder.function("main", (source, extra), (*results, reused))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("nested", (False, True))
def test_internal_device_completion_discharges_caller_war(nested):
    module = completed_read_graph(nested=nested)
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert any(event.function == "worker" and event.before == "completed"
               and event.scope == "device" for event in events)
    assert not any(event.function == "main" and event.before == "reused"
                   for event in events)


@pytest.mark.parametrize("completion_scope,tail_read", (("chip", False), ("block", False), ("device", True)))
def test_incomplete_or_tail_peer_read_remains_pending(completion_scope, tail_read):
    module = completed_read_graph(completion_scope=completion_scope, tail_read=tail_read)
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert any(event.function == "main" and event.before == "reused"
               and event.scope == "device" for event in events)


def test_caller_peer_write_cannot_use_a_callee_local_completion():
    module = completed_read_graph(read_scope="chip", completion_scope="chip", write_scope="device")
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert any(event.function == "worker" and event.before == "completed"
               and event.scope == "grid" for event in events)
    assert any(event.function == "main" and event.before == "reused"
               and event.scope == "device" for event in events)


@pytest.mark.parametrize("prior_caller_read,tail_read,second_call", ((True, False, False), (False, True, True)))
def test_callee_completion_also_covers_earlier_caller_accesses(prior_caller_read, tail_read, second_call):
    module = completed_read_graph(prior_caller_read=prior_caller_read,
                                  tail_read=tail_read, second_call=second_call)
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert not any(event.function == "main" and event.before == "reused" for event in events)


def test_summary_is_independent_of_function_declaration_order():
    from dataclasses import replace
    module = completed_read_graph(nested=True)
    module = replace(module, functions=tuple(reversed(module.functions)))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert not any(event.function == "main" and event.before == "reused" for event in events)


@pytest.mark.parametrize("index", (0, 1, 2, None))
@pytest.mark.parametrize("tail_read", (False, True))
def test_call_exit_preserves_reference_slice_ranges(index, tail_read):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    tensor = fm.tensor_type("float32", (1, 64))
    state_type = fm.RefType("state", (("cache", fm.tensor_type("float32", (3, 64))),))
    view_type = fm.RefType("state", (("cache", tensor),))
    scalar = fm.tensor_type("int32", ())
    remote = _kernel("remote_ref", view_type, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE), tensor)
    completion = _kernel("completion_ref", tensor, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE))
    writer = _kernel("writer_ref", view_type, fm.MemoryEffect.WRITE, tensor)
    for primitive in (remote, completion, writer):
        builder.prim_function(primitive)
    formal = builder.var("formal_state", state_type, id="formal_state")
    layer = (builder.var("layer", scalar, id="layer") if index is None else builder.call(
        "tir.scalar_const", (), scalar, id="layer", attrs={"value": index}))
    view = builder.call("tir.ref_slice", (formal, layer), view_type,
                        id="view", attrs={"length": 1})
    read = builder.call("tir.call", (view,), tensor, id="ref_read", attrs={"callee": remote.name})
    completed = builder.call("tir.call", (read,), tensor, id="ref_completed",
                             attrs={"callee": completion.name})
    outputs = [completed]
    if tail_read:
        outputs.append(builder.call("tir.call", (view,), tensor, id="ref_tail", attrs={"callee": remote.name}))
    parameters = (formal, layer) if index is None else (formal,)
    builder.function("worker", parameters, outputs)
    source = builder.var("state", state_type, id="state")
    one = builder.call("tir.scalar_const", (), scalar, id="one", attrs={"value": 1})
    arguments = (source, one) if index is None else (source,)
    result_type = tensor if len(outputs) == 1 else fm.TupleType(tuple(tensor for _ in outputs))
    invocation = builder.call("tir.call", arguments, result_type,
                              id="ref_invocation", attrs={"callee": "worker"})
    selected = builder.call("tir.ref_slice", (source, one), view_type, id="selected", attrs={"length": 1})
    reused = builder.call("tir.call", (selected,), tensor, id="ref_reused", attrs={"callee": writer.name})
    builder.function("main", (source,), (invocation, reused))
    module = fm.verify_module(builder.build(entry="main"))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    boundary = [event for event in events if event.function == "main" and event.before == "ref_reused"]
    assert bool(boundary) is (tail_read and index in (1, None))
    if boundary:
        assert boundary[0].scope == "device"
        assert all(span.offset == 256 and span.nbytes == 256 for span in boundary[0].ranges)


@pytest.mark.parametrize("alias_arguments", (False, True))
def test_tail_effect_is_instantiated_for_each_actual_alias(alias_arguments):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    tensor = fm.tensor_type("float32", (1, 64))
    remote = _kernel("remote_alias", tensor, fm.MemoryEffect(
        fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE))
    writer = _kernel("writer_alias", tensor, fm.MemoryEffect.WRITE)
    builder.prim_function(remote)
    builder.prim_function(writer)
    first = builder.var("first", tensor, id="first")
    second = builder.var("second", tensor, id="second")
    read = builder.call("tir.call", (first,), tensor, id="alias_read", attrs={"callee": remote.name})
    completed = builder.call("tir.call", (read,), tensor, id="alias_completed", attrs={"callee": remote.name})
    tail = builder.call("tir.call", (second,), tensor, id="alias_tail", attrs={"callee": remote.name})
    builder.function("worker", (first, second), (completed, tail))
    source = builder.var("source", tensor, id="source")
    extra = builder.var("extra", tensor, id="extra")
    invocation = builder.call("tir.call", (source, source if alias_arguments else extra),
                              fm.TupleType((tensor, tensor)), id="alias_invocation", attrs={"callee": "worker"})
    reused = builder.call("tir.call", (source,), tensor, id="alias_reused", attrs={"callee": writer.name})
    builder.function("main", (source, extra), (invocation, reused))
    module = fm.verify_module(builder.build(entry="main"))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    boundary = [event for event in plan_memory_synchronization(module, plan).events
                if event.function == "main" and event.before == "alias_reused"]
    assert bool(boundary) is alias_arguments
    if boundary:
        assert boundary[0].scope == "device"
