# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reusable functions must expose the accesses made by their bodies."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.bufferization import MemorySharingScope
from triton.flagmega.passes.tir.bufferize import plan_memory_synchronization
from triton.flagmega.passes.tir.bufferize.call_accesses import CallAccessResolver
from triton.flagmega.passes.tir.bufferize.synchronization import _node_accesses
from .test_memory_synchronization import _workspace_options


def call_graph(*, collective, nested=False, unrelated=False):
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    tensor = fm.tensor_type("float32", (1, 64))
    distributed = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1))),
                                     fm.Placement((2, 4), "yx", "bb"))
    formal = builder.var("formal", distributed, id="formal")
    extra = builder.var("extra", distributed, id="extra")
    local = builder.call("tir.kernel", (formal, ), distributed, id="local")
    collected = builder.call("tir.kernel", (extra if unrelated else formal, ), distributed, id="collected",
                             attrs={"semantic_op": "distributed.boxing"} if collective else {})
    builder.function("worker", (formal, extra), (local, collected))
    callee = "worker"
    if nested:
        outer = builder.var("outer", distributed, id="outer")
        extra_outer = builder.var("extra_outer", distributed, id="extra_outer")
        forwarded = builder.call("tir.call", (outer, extra_outer), fm.TupleType((distributed, distributed)),
                                 id="forwarded", attrs={"callee": callee})
        first = builder.call("builtin.get_item", (forwarded, ), distributed, id="first", attrs={"index": 0})
        second = builder.call("builtin.get_item", (forwarded, ), distributed, id="second", attrs={"index": 1})
        builder.function("wrapper", (outer, extra_outer), (first, second))
        callee = "wrapper"
    source = builder.var("source", distributed, id="source")
    other = builder.var("other", distributed, id="other")
    produced = builder.call("tir.kernel", (source, ), distributed, id="produced")
    call = builder.call("tir.call", (produced, other), fm.TupleType((distributed, distributed)), id="invocation",
                        attrs={"callee": callee})
    builder.function("main", (source, other), (call, ))
    return builder.build(entry="main")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("collective,unrelated,scope", [(True, False, "grid"), (False, False, "block"),
                                                        (True, True, "block")])
def test_call_boundary_uses_per_operand_callee_accesses(nested, collective, unrelated, scope):
    module = call_graph(collective=collective, nested=nested, unrelated=unrelated)
    plan = fm.make_buffer_plan(module, options=_workspace_options(MemorySharingScope.CHIP))
    events = plan_memory_synchronization(module, plan).events
    boundary = next(event for event in events if event.function == "main" and event.before == "invocation")
    assert boundary.scope == scope
    assert "WRITE->READ" in boundary.hazards


@pytest.mark.parametrize("index", [0, 1, 2, None])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_callee_reference_slice_keeps_its_nonzero_or_bounded_byte_span(index, level):
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.options import CompileOptions
    from .test_ref_slice import state_slice_graph

    module = Compiler(CompileOptions(bufferize_opt_level=level)).compile(state_slice_graph(index, reusable=True)).module
    plan = fm.verify_buffer_plan(module)
    assert plan.optimization_level == level
    node = module.node_map["first"]
    bindings = dict(plan.function_map["main"].values)
    accesses = CallAccessResolver(module, plan, _node_accesses).accesses("main", node, bindings)
    state_id = dict(plan.function_map["main"].parameters)["entry_state"][0]
    descriptor = plan.buffer_map[state_id]
    state_accesses = [access for access in accesses if access.buffer == state_id]
    assert state_accesses
    layer_bytes = descriptor.nbytes // 3
    for access in state_accesses:
        assert access.is_reference
        assert access.offset == (descriptor.offset if index is None else descriptor.offset + index * layer_bytes)
        assert access.nbytes == (descriptor.nbytes if index is None else layer_bytes)
