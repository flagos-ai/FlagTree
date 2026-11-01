# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit distributed call/state interfaces must survive automatic layout."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.core import construction_scope
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target


def owned_call_graph(*, explicit=True, tuple_output=False):
    mesh = fm.Placement((4, 2, 2), "cyx", "cbb")
    tensor = fm.tensor_type("float32", (16,))
    state_type = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0,)),), mesh) if explicit else tensor
    auxiliary = fm.tensor_type("float32", (8,))

    class Graph(fm.Module):
        def forward(self):
            state = self.input("state", state_type, id="state")
            next_state = fm.F.math.add(state, state, name="next_state")
            first = self.input("first", state_type, id="first")
            if tuple_output:
                extra = self.input("extra", auxiliary, id="extra")
                second = self.input("second", auxiliary, id="second")
                outputs = (next_state, extra)
                result_type = fm.TupleType(tuple(value.type for value in outputs))
                call = fm.F.builtin.call(first, second, result_type=result_type, callee="step", name="call")
                self.function("step", (state, extra), outputs)
                self.function("main", (first, second), (call,))
            else:
                call = fm.F.builtin.call(first, result_type=next_state.type, callee="step", name="call")
                self.function("step", (state,), (next_state,))
                self.function("main", (first,), (call,))

    module = Graph(dialect="ntt", stage="packed", entry="main").build()
    target = NvidiaSm90Target(options=replace(NvidiaSm90Target().options, placements=(mesh,)))
    return module, target, state_type


@pytest.mark.parametrize("tuple_output", (False, True))
def test_explicit_state_ownership_survives_call_and_program_boundary(tmp_path, tuple_output):
    source, target, state_type = owned_call_graph(tuple_output=tuple_output)
    module = AutoDistributedPass.run(source, target)
    main = module.function_map["main"]
    output_type = module.node_map[main.outputs[0]].type
    actual = output_type.fields[0] if tuple_output else output_type
    assert actual == state_type
    call_type = module.node_map["call"].type
    assert (call_type.fields[0] if tuple_output else call_type) == state_type
    assert module.node_map[module.function_map["step"].parameters[0]].type == state_type
    for node in module.nodes:
        if node.op not in {"distributed.boxing", "distributed.force_boxing"}:
            continue
        before = module.node_map[node.inputs[0]].type
        if isinstance(before, fm.DistributedType) and isinstance(node.type, fm.DistributedType):
            assert not fm.boxing_crossing_chip_axes(before, node.type)
    assert fm.load_module(fm.emit_module(module, tmp_path / "owned.py")).semantic_hash == module.semantic_hash


def test_unannotated_call_interface_keeps_the_existing_broadcast_default():
    source, target, _ = owned_call_graph(explicit=False)
    module = AutoDistributedPass.run(source, target)
    output = module.node_map[module.function_map["main"].outputs[0]].type
    assert isinstance(output, fm.DistributedType)
    assert all(policy == fm.SBP.broadcast() for policy in output.axis_policies)


@pytest.mark.parametrize("trips", (0, 3))
def test_owned_state_call_remains_sharded_through_for_iter_args(tmp_path, trips):
    mesh = fm.Placement((4, 2, 2), "cyx", "cbb")
    state_type = fm.DistributedType(fm.tensor_type("float32", (16,)),
                                    (fm.SBP.split_contiguous((0,)),), mesh)
    index_type = fm.tensor_type("int64", ())
    builder = fm.IRBuilder(dialect="ntt", stage="packed")
    state = builder.var("state", state_type, id="state")
    with construction_scope(builder):
        updated = fm.F.math.add(state, state, name="updated")
    builder.function("step", (state,), (updated,))
    iv = builder.var("iv", index_type, id="iv")
    carried = builder.var("carried", state_type, id="carried")
    with construction_scope(builder):
        call = fm.F.builtin.call(carried, result_type=state_type, callee="step", name="call")
    body = builder.region("body", (iv, carried), (call,), iter_count=1)
    source = builder.var("input", state_type, id="input")
    with construction_scope(builder):
        lower = fm.F.builtin.scalar_const(index_type, 0)
        upper = fm.F.builtin.scalar_const(index_type, trips)
        stride = fm.F.builtin.scalar_const(index_type, 1)
        loop = fm.F.control.for_loop(lower, upper, stride, source, body=body, name="loop")
        output = fm.F.tensors.get_item(loop, 0)
    builder.function("main", (source,), (output,))
    module = fm.verify_module(builder.build(entry="main"))
    target = NvidiaSm90Target(options=replace(NvidiaSm90Target().options, placements=(mesh,)))
    distributed = AutoDistributedPass.run(module, target)
    assert distributed.node_map["loop"].type.fields[0] == state_type
    assert distributed.node_map["carried"].type == state_type
    assert distributed.node_map["call"].type == state_type
    assert fm.load_module(fm.emit_module(distributed, tmp_path / "owned-loop.py")).semantic_hash == distributed.semantic_hash


def test_parameter_bank_element_keeps_mixed_owned_and_logical_fields(tmp_path):
    mesh = fm.Placement((4, 2, 2), "cyx", "cbb")
    owned = fm.DistributedType(fm.tensor_type("float32", (16,)), (fm.SBP.split_contiguous((0,)),), mesh)
    logical = fm.tensor_type("float32", (8,))
    bank_type = fm.RefType("parameter_bank", (("element", fm.TupleType((owned, logical))),))

    class Graph(fm.Module):
        def forward(self):
            state = self.input("state", owned, id="state")
            extra = self.input("extra", logical, id="extra")
            result = fm.F.math.add(state, state)
            self.function("step", (state, extra), (result, extra))
            bank = self.input("bank", bank_type, id="bank")
            index = fm.F.builtin.scalar_const(fm.tensor_type("int64", ()), 0)
            row = fm.F.control.bank_get(bank, index, name="row")
            first, second = (fm.F.tensors.get_item(row, slot) for slot in (0, 1))
            call = fm.F.builtin.call(first, second, callee="step",
                                     result_type=fm.TupleType((owned, logical)), name="call")
            self.function("main", (bank,), (call,))

    target = NvidiaSm90Target(options=replace(NvidiaSm90Target().options, placements=(mesh,)))
    module = AutoDistributedPass.run(Graph(dialect="ntt", stage="packed", entry="main").build(), target)
    row_type = module.node_map["row"].type
    assert row_type.fields[0] == owned
    assert all(policy == fm.SBP.broadcast() for policy in row_type.fields[1].axis_policies)
    assert dict(module.node_map["bank"].type.fields)["element"] == row_type
    assert module.node_map["call"].type.fields[0] == owned
    assert fm.load_module(fm.emit_module(module, tmp_path / "owned-bank.py")).semantic_hash == module.semantic_hash


def test_explicit_interface_cannot_be_reinterpreted_on_another_mesh():
    module, _, _ = owned_call_graph()
    target = NvidiaSm90Target(options=replace(NvidiaSm90Target().options,
                                             placements=(fm.Placement((4, 4), "yx", "bb"),)))
    with pytest.raises(IRVerificationError, match="active placement"):
        AutoDistributedPass.run(module, target)
