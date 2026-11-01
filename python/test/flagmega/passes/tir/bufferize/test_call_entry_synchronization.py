# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real typed call frames: late peer reads have an earlier internal fence."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization

from python.test.flagmega.passes.tir.bufferize.synchronization.test_axis_group_inference import _chip_workspace_options
from python.test.flagmega.passes.tir.bufferize.test_call_exit_synchronization import _kernel


def fixture(*, peer_first=False, nested=False, prior_peer=False, write_after_fence=False):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    mesh = fm.Placement((2, 4), "cx", "cb")
    kind = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)), (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), mesh
    )
    writer = _kernel("local_writer", kind, fm.MemoryEffect(fm.MemoryAccessMode.WRITE, fm.MemoryAccessScope.CHIP))
    peer = _kernel("peer_reader", kind, fm.MemoryEffect(fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE))
    builder.prim_function(writer)
    builder.prim_function(peer)
    formal = builder.var("formal", kind, id="formal")
    values = []
    for name, callee in (
        (("peer", "peer_reader"), ("local", "local_writer"))
        if peer_first
        else (("local", "local_writer"), ("peer", "peer_reader"))
    ):
        values.append(builder.call("tir.call", (formal,), kind, id="worker_" + name, attrs={"callee": callee}))
    if write_after_fence:
        values.append(
            builder.call("tir.call", (formal,), kind, id="worker_later_write", attrs={"callee": "local_writer"})
        )
        values.append(
            builder.call("tir.call", (formal,), kind, id="worker_later_peer", attrs={"callee": "peer_reader"})
        )
    builder.function("worker", (formal,), values)
    result_kind = fm.TupleType(tuple(kind for _ in values))
    callee = "worker"
    if nested:
        outer = builder.var("outer", kind, id="outer")
        called = builder.call("tir.call", (outer,), result_kind, id="nested", attrs={"callee": "worker"})
        builder.function("wrapper", (outer,), (called,))
        callee = "wrapper"
    source = builder.var("source", kind, id="source")
    produced = builder.call("tir.call", (source,), kind, id="producer", attrs={"callee": "local_writer"})
    previous = (
        builder.call("tir.call", (produced,), kind, id="previous_peer", attrs={"callee": "peer_reader"})
        if prior_peer
        else None
    )
    invoked = builder.call("tir.call", (produced,), result_kind, id="invocation", attrs={"callee": callee})
    builder.function("main", (source,), ((previous, invoked) if previous is not None else (invoked,)))
    module = fm.verify_module(builder.build(entry="main"))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    return module, plan


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("peer_first", (False, True))
def test_only_an_internal_fence_before_the_peer_access_discharges_the_entry_device_hazard(peer_first, nested):
    module, plan = fixture(peer_first=peer_first, nested=nested)
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "worker" and e.scope == "device" for e in events)
    entry = [e for e in events if e.function == "main" and e.before == "invocation"]
    assert any(e.scope == "device" for e in entry) is peer_first
    if not peer_first:
        # The earlier local write still needs ordinary chip ordering. The
        # late peer read is already published by the worker's own fence.
        assert all(e.scope != "device" for e in entry)


@pytest.mark.parametrize("nested", (False, True))
def test_a_late_fence_cannot_protect_an_early_write_from_a_previous_peer_reader(nested):
    module, plan = fixture(prior_peer=True, nested=nested)
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "main" and e.before == "invocation" and e.scope == "device" for e in events)


@pytest.mark.parametrize("nested", (False, True))
def test_entry_coverage_cannot_publish_a_newer_local_write_inside_the_same_callee(nested):
    module, plan = fixture(write_after_fence=True, nested=nested)
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "worker" and e.before == "worker_later_peer" and e.scope == "device" for e in events)


@pytest.mark.parametrize("peer_first", (False, True))
def test_entry_coverage_is_independent_of_function_declaration_order(peer_first):
    module, _ = fixture(peer_first=peer_first, nested=True)
    module = replace(module, functions=tuple(reversed(module.functions)))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "main" and e.before == "invocation" and e.scope == "device" for e in events) is peer_first


def separated_effects_fixture(*, local_scope="chip", loop=False, nested=False):
    """An unrelated local fence or possibly empty loop cannot publish B."""
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    mesh = fm.Placement((2, 4), "cx", "cb")
    kind = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)), (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))), mesh
    )
    scalar = fm.tensor_type("int64", ())
    writer = _kernel(
        "separate_writer", kind, fm.MemoryEffect(fm.MemoryAccessMode.WRITE, fm.MemoryAccessScope(local_scope))
    )
    reader = _kernel(
        "separate_reader", kind, fm.MemoryEffect(fm.MemoryAccessMode.READ, fm.MemoryAccessScope(local_scope))
    )
    peer = _kernel("separate_peer", kind, fm.MemoryEffect(fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE))
    for primitive in (writer, reader, peer):
        builder.prim_function(primitive)
    if loop:
        iv = builder.var("iv", scalar, id="iv")
        carried = builder.var("carried", kind, id="carried")
        local = builder.call("tir.call", (carried,), kind, id="body_local", attrs={"callee": writer.name})
        remote = builder.call("tir.call", (local,), kind, id="body_peer", attrs={"callee": peer.name})
        builder.region("body", (iv, carried), (remote,), iter_count=1)
    a = builder.var("a", kind, id="a")
    b = builder.var("b", kind, id="b")
    parameters = [a, b]
    if loop:
        bounds = tuple(builder.var(name, scalar, id=name) for name in ("lower", "upper"))
        parameters.extend(bounds)
        step = builder.call("tir.scalar_const", (), scalar, id="positive_step", attrs={"value": 1})
        repeated = builder.call(
            "control.for",
            (*bounds, step, a),
            fm.TupleType((kind,)),
            id="maybe_empty",
            attrs={"callee": "body", "iter_count": 1, "effect_kind": "pure", "effect_resource": None},
        )
        local = builder.call("builtin.get_item", (repeated,), kind, id="loop_result", attrs={"index": 0})
        before_peer = local
    else:
        local = builder.call("tir.call", (a,), kind, id="separate_local", attrs={"callee": writer.name})
        before_peer = builder.call("tir.call", (a,), kind, id="separate_read", attrs={"callee": reader.name})
    late = builder.call("tir.call", (b,), kind, id="separate_late_peer", attrs={"callee": peer.name})
    builder.function("separate_worker", parameters, (local, before_peer, late))
    result_kind = fm.TupleType((kind, kind, kind))
    callee = "separate_worker"
    kinds = [kind, kind] + ([scalar] * 2 if loop else [])
    if nested:
        outer = tuple(builder.var("outer" + str(i), t, id="outer" + str(i)) for i, t in enumerate(kinds))
        wrapped = builder.call("tir.call", outer, result_kind, id="separate_nested", attrs={"callee": callee})
        builder.function("separate_wrapper", outer, (wrapped,))
        callee = "separate_wrapper"
    inputs = tuple(builder.var("source" + str(i), t, id="source" + str(i)) for i, t in enumerate(kinds))
    produced = builder.call("tir.call", (inputs[1],), kind, id="separate_producer", attrs={"callee": writer.name})
    invoked = builder.call(
        "tir.call", (inputs[0], produced, *inputs[2:]), result_kind, id="separate_invocation", attrs={"callee": callee}
    )
    builder.function("main", inputs, (invoked,))
    module = fm.verify_module(builder.build(entry="main"))
    return module, fm.make_buffer_plan(module, options=_chip_workspace_options())


@pytest.mark.parametrize("scope", ("block", "chip"))
@pytest.mark.parametrize("nested", (False, True))
def test_local_completion_does_not_cover_a_later_peer_access_to_another_input(scope, nested):
    module, plan = separated_effects_fixture(local_scope=scope, nested=nested)
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "separate_worker" and e.before == "separate_read" for e in events)
    assert not any(e.function == "separate_worker" and e.scope == "device" for e in events)
    assert any(e.function == "main" and e.before == "separate_invocation" and e.scope == "device" for e in events)


@pytest.mark.parametrize("nested", (False, True))
def test_possible_zero_trip_region_does_not_export_its_internal_device_fence(nested):
    module, plan = separated_effects_fixture(loop=True, nested=nested)
    events = plan_memory_synchronization(module, plan).events
    assert any(e.function == "body" and e.scope == "device" for e in events)
    assert any(e.function == "main" and e.before == "separate_invocation" and e.scope == "device" for e in events)
