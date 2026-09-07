# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase GenerateBoxingValue semantics, independent of target kernels."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _graph(op="distributed.boxing", *, literal=False, identity=False):
    tensor = fm.tensor_type("float32", (2, 259))
    placement = fm.Placement((8, 16), "yx", "bb")
    local = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)), placement)
    partial = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()),
                                 placement, fm.SBP.partial((0, 1), fm.ReduceOp.SUM))
    source_type = fm.TupleType((local, fm.TupleType((tensor, partial))))
    target_type = source_type if identity else fm.TupleType((tensor, fm.TupleType((local, partial))))
    builder = fm.IRBuilder(dialect="distributed", stage="gather_reduce_qkv_fused")
    if literal:
        a, b, c = (builder.var(name, typ, id=name) for name, typ in (("a", local), ("b", tensor), ("c", partial)))
        nested = builder.call("builtin.tuple", (b, c), source_type.fields[1], id="nested")
        source = builder.call("builtin.tuple", (a, nested), source_type, id="source")
        parameters = (a, b, c)
    else:
        source = builder.var("source", source_type, id="source")
        parameters = (source,)
    result = builder.call(op, (source,), target_type, id="result", attrs={"new_type": target_type})
    builder.function("main", parameters, (result,))
    return fm.verify_module(builder.build(entry="main"))


def _lower(module):
    return Compiler().run_stage(module, "lower-tuple-boxing").module


@pytest.mark.parametrize("op", ("distributed.boxing", "distributed.force_boxing"))
@pytest.mark.parametrize("literal", (False, True))
def test_mixed_transfers_are_scalar_leaves_and_partial_identity_is_an_alias(op, literal):
    before = _graph(op, literal=literal)
    after = _lower(before)
    transfers = [node for node in after.nodes if node.op == op]
    assert len(transfers) == 2
    assert all(not isinstance(node.type, fm.TupleType) for node in transfers)
    assert {isinstance(node.type, fm.DistributedType) for node in transfers} == {False, True}
    assert all(not isinstance(node.type, fm.DistributedType) or node.type.partial is None for node in transfers)
    assert after.node_map["result"].op == "builtin.tuple"
    assert after.functions == before.functions
    if literal:
        nested = after.node_map[after.node_map["result"].inputs[1]]
        assert nested.inputs[1] == "c"
        assert not any(node.op == "builtin.get_item" for node in after.nodes)
    fm.verify_module(after)


def test_whole_identity_tuple_does_not_allocate_or_transfer():
    after = _lower(_graph(identity=True))
    assert after.node_map["result"].op == "builtin.tuple"
    assert all(after.node_map[value].op == "builtin.get_item" for value in after.node_map["result"].inputs)
    assert not any(node.op.startswith("distributed.") for node in after.nodes)


def test_lowering_is_idempotent_and_editable(tmp_path):
    first = _lower(_graph())
    second = _lower(replace(first, stage="gather_reduce_qkv_fused"))
    assert second.nodes == first.nodes
    assert fm.load_module(fm.emit_module(first, tmp_path / "lowered.py")) == first


def test_generated_names_do_not_collide_with_existing_nodes():
    module = _graph(literal=True)
    # This name deliberately occupies the first leaf's natural generated ID.
    conflict = fm.Node("result.field_0", "builtin.scalar_const", (), fm.tensor_type("int32", ()), attrs={"value": 1})
    module = replace(module, nodes=(conflict, *module.nodes))
    result = _lower(module)
    assert result.node_map[conflict.id] == conflict
    assert len(result.nodes) == len({node.id for node in result.nodes})
