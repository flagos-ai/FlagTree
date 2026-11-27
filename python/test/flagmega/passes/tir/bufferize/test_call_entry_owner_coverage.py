# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed incoming coverage across partial-owner and offset reference call frames."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization

from python.test.flagmega.passes.tir.bufferize.synchronization.test_axis_group_inference import (
    _chip_workspace_options,
)
from python.test.flagmega.passes.tir.bufferize.synchronization.test_axis_group_inference import (
    _kernel as read_kernel,
)
from python.test.flagmega.passes.tir.bufferize.test_call_exit_synchronization import _kernel


def planned(builder):
    module = fm.verify_module(builder.build(entry="main"))
    if any(isinstance(n.type, fm.RefType) for n in module.nodes):
        mesh = fm.Placement((2, 4), "cx", "cb")
        module = fm.verify_module(replace(module, metadata={"launch_contract": {"grid_mesh": mesh.to_data()}}))
    plan = fm.make_buffer_plan(module, options=_chip_workspace_options())
    return module, plan, plan_memory_synchronization(module, plan).events


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("partial_axes", ((1,), (0,)))
def test_partial_owner_barrier_covers_only_its_actual_scope(nested, partial_axes):
    b = fm.IRBuilder(dialect="tir", stage="selected_tir")
    mesh = fm.Placement((2, 2, 2), "cyx", "cbb")
    kind = fm.DistributedType(
        fm.tensor_type("float32", (1, 64)),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((2,), 8)),
        mesh,
        partial=fm.SBP.partial(partial_axes),
    )
    local = read_kernel("partial_local", kind, fm.MemoryEffect.READ)
    group = read_kernel("partial_group", kind, fm.MemoryEffect.READ.across_partial_owners())
    peer = read_kernel("partial_peer", kind, fm.MemoryEffect(fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE))
    for k in (local, group, peer):
        b.prim_function(k)
    a = b.var("a", kind, id="a")
    other = b.var("b", kind, id="b")
    produced = b.call("tir.call", (a,), kind, id="worker_local", attrs={"callee": local.name})
    reduced = b.call("tir.call", (produced,), kind, id="worker_group", attrs={"callee": group.name})
    late = b.call("tir.call", (other,), kind, id="worker_peer", attrs={"callee": peer.name})
    result_kind = fm.TupleType((kind, kind))
    b.function("worker", (a, other), (reduced, late))
    callee = "worker"
    if nested:
        x = b.var("x", kind, id="x")
        y = b.var("y", kind, id="y")
        invoked = b.call("tir.call", (x, y), result_kind, id="wrapped", attrs={"callee": callee})
        b.function("wrapper", (x, y), (invoked,))
        callee = "wrapper"
    srca = b.var("srca", kind, id="srca")
    srcb = b.var("srcb", kind, id="srcb")
    made = b.call("tir.call", (srcb,), kind, id="caller_local", attrs={"callee": local.name})
    call = b.call("tir.call", (srca, made), result_kind, id="invocation", attrs={"callee": callee})
    b.function("main", (srca, srcb), (call,))
    _, _, events = planned(b)
    boundary = [e for e in events if e.function == "main" and e.before == "invocation" and e.scope == "device"]
    if partial_axes == (1,):
        assert any(
            e.function == "worker" and e.before == "worker_group" and e.scope == "grid" and e.axis_group_axes == (1,)
            for e in events
        )
        assert not any(e.function == "worker" and e.scope == "device" for e in events)
        assert boundary, "A block-axis group cannot publish the unrelated B input to peer chips"
    else:
        assert any(e.function == "worker" and e.before == "worker_group" and e.scope == "device" for e in events)
        assert not boundary, "The guaranteed device barrier precedes the B peer access"


@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("peer_first", (False, True))
@pytest.mark.parametrize("row", (0, 1, 2))
def test_nonzero_reference_slice_preserves_incoming_coverage_and_bound_bytes(nested, peer_first, row):
    b = fm.IRBuilder(dialect="tir", stage="selected_tir")
    state = fm.RefType("state", (("cache", fm.tensor_type("float32", (3, 64))),))
    view = fm.RefType("state", (("cache", fm.tensor_type("float32", (1, 64))),))
    result = fm.tensor_type("float32", (1, 64))
    scalar = fm.tensor_type("int32", ())
    write = _kernel("ref_write", view, fm.MemoryEffect(fm.MemoryAccessMode.WRITE, fm.MemoryAccessScope.CHIP), result)
    read = _kernel("ref_peer", view, fm.MemoryEffect(fm.MemoryAccessMode.READ, fm.MemoryAccessScope.DEVICE), result)
    b.prim_function(write)
    b.prim_function(read)
    formal = b.var("formal", state, id="formal")
    index = b.call("tir.scalar_const", (), scalar, id="worker_row", attrs={"value": row})
    selected = b.call("tir.ref_slice", (formal, index), view, id="worker_slice", attrs={"length": 1})
    order = (read, write) if peer_first else (write, read)
    results = tuple(
        b.call("tir.call", (selected,), result, id="worker_" + k.name, attrs={"callee": k.name}) for k in order
    )
    b.function("worker", (formal,), results)
    callee = "worker"
    returns = fm.TupleType((result, result))
    if nested:
        outer = b.var("outer", state, id="outer")
        call = b.call("tir.call", (outer,), returns, id="wrapped", attrs={"callee": "worker"})
        b.function("wrapper", (outer,), (call,))
        callee = "wrapper"
    source = b.var("source", state, id="source")
    at = b.call("tir.scalar_const", (), scalar, id="caller_row", attrs={"value": row})
    actual = b.call("tir.ref_slice", (source, at), view, id="caller_slice", attrs={"length": 1})
    made = b.call("tir.call", (actual,), result, id="caller_write", attrs={"callee": write.name})
    invoked = b.call("tir.call", (source,), returns, id="invocation", attrs={"callee": callee})
    b.function("main", (source,), (made, invoked))
    _, _, events = planned(b)
    boundary = [e for e in events if e.function == "main" and e.before == "invocation" and e.scope == "device"]
    assert bool(boundary) is peer_first
    if boundary:
        assert any(span.offset == row * 256 and span.nbytes == 256 for e in boundary for span in e.ranges)
    assert any(e.function == "worker" and e.scope == "device" for e in events)
