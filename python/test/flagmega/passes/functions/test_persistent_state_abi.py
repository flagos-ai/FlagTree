# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit carried-state ownership removes only proven boundary restoration."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.functions.persistent_state_abi import specialize_persistent_state_abi


def state_module(*, observer=False, exported=False, mismatched=False, packed=False):
    b = fm.IRBuilder(dialect="distributed", stage="distributed")
    mesh = fm.Placement((2, 2, 4), "cyx", "cbb")
    dtype = fm.VectorType(fm.DType.FLOAT32, (2,)) if packed else fm.DType.FLOAT32
    logical = fm.tensor_type(dtype, (1, 8, 4))
    broad = fm.SBP.broadcast()
    owned = fm.DistributedType(logical, (broad, fm.SBP.split_contiguous((0, 2), 1), broad), mesh)
    replicated = fm.DistributedType(logical, (broad,) * 3, mesh)
    alternative = fm.DistributedType(logical, (broad, fm.SBP.split_contiguous((1, 2), 1), broad), mesh)
    p, q = (b.var(name, owned, id=name) for name in ("p", "q"))
    p_out = b.call("math.add", (p, p), owned, id="p_out")
    q_out = b.call("math.add", (q, q), owned, id="q_out")
    b.function("step", (p, q), (p_out, q_out), attrs={"reusable": True})
    parameters, results, pairs = [], [], []
    for layer in range(2):
        arguments = []
        for field in range(2):
            name = f"state{layer}_{field}"
            value = b.var(name, logical, id=name)
            parameters.append(value)
            copied = b.call("distributed.boxing", (value,), replicated,
                            attrs={"new_type": replicated}, id=f"{name}_copy")
            shard = b.call("distributed.sharded_view", (copied,), owned,
                           attrs={"new_type": owned}, id=f"{name}_shard")
            arguments.append(shard)
            if layer == field == 0 and exported:
                results.append(copied)
            if layer == field == 0 and observer:
                results.append(b.call("math.add", (copied, copied), replicated, id="observer"))
        call = b.call("builtin.call", arguments, fm.TupleType((owned, owned)),
                      attrs={"callee": "step"}, id=f"call{layer}")
        for field in range(2):
            item = b.call("builtin.get_item", (call,), owned, attrs={"index": field}, id=f"result{layer}_{field}")
            target = alternative if mismatched and layer == field == 0 else replicated
            out = b.call("distributed.boxing", (item,), target,
                         attrs={"new_type": target}, id=f"output{layer}_{field}")
            results.append(out)
            pairs.append((parameters[layer * 2 + field].id, out.id))
    b.function("main", parameters, results)
    abi = {"schema": "flagmega.composite-abi/v1", "tensor_states": tuple(
        {"state_id": layer, "input": tuple(pair[0] for pair in pairs[layer * 2:layer * 2 + 2]),
         "output": tuple(pair[1] for pair in pairs[layer * 2:layer * 2 + 2])} for layer in range(2))}
    source = replace(b.build(entry="main"), metadata={"auto_distribution": {"placement": mesh.to_data()},
                                                     "composite_abi": abi})
    return fm.verify_module(source), tuple(pairs)


@pytest.mark.parametrize("packed", [False, True])
def test_owned_state_abi_eliminates_boundary_copies_and_roundtrips(tmp_path, packed):
    source, pairs = state_module(packed=packed)
    assert sum(node.op == "distributed.boxing" for node in source.nodes) == 8
    result = specialize_persistent_state_abi(source, pairs)
    assert not any(node.op in {"distributed.boxing", "distributed.sharded_view"} for node in result.nodes)
    assert result.function_map["step"] is source.function_map["step"]
    for layer in range(2):
        assert result.node_map[f"call{layer}"].inputs == tuple(pair[0] for pair in pairs[layer * 2:layer * 2 + 2])
    for parameter, public in pairs:
        owned_result = public.replace("output", "result")
        assert result.node_map[parameter].type == source.node_map[owned_result].type
    assert all(value in result.function_map["main"].outputs
               for state in result.metadata["composite_abi"]["tensor_states"] for value in state["output"])
    updated_pairs = tuple(zip(result.function_map["main"].parameters, result.function_map["main"].outputs, strict=True))
    assert specialize_persistent_state_abi(result, updated_pairs).semantic_hash == result.semantic_hash
    path = fm.emit_module(result, tmp_path / "owned.py")
    assert fm.load_module(path).semantic_hash == result.semantic_hash


def test_complete_logical_reference_and_repeated_steps_are_unchanged():
    source, pairs = state_module()
    result = specialize_persistent_state_abi(source, pairs)
    values = {pair[0]: (torch.arange(32).reshape(1, 8, 4).float() + index) / 16
              for index, pair in enumerate(pairs)}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    for _ in range(4):
        expected = evaluator.run(source, values)
        actual = evaluator.run(result, values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        values = {pair[0]: value for pair, value in zip(pairs, actual, strict=True)}


@pytest.mark.parametrize("observer,exported,mismatched", [(True, False, False), (False, True, False), (False, False, True)])
def test_shared_full_observers_and_different_ownership_reject(observer, exported, mismatched):
    source, pairs = state_module(observer=observer, exported=exported, mismatched=mismatched)
    with pytest.raises(IRVerificationError, match="observer|different ownership"):
        specialize_persistent_state_abi(source, pairs)


def test_duplicate_input_and_wrong_entry_boundary_reject():
    source, pairs = state_module()
    with pytest.raises(IRVerificationError, match="distinct"):
        specialize_persistent_state_abi(source, (pairs[0], pairs[0]))
    with pytest.raises(IRVerificationError, match="entry parameter"):
        specialize_persistent_state_abi(source, (("p", pairs[0][1]),))


def test_changed_boundary_decisions_are_invalidated_and_callee_decisions_retained():
    source, pairs = state_module()
    owners = (pairs[0][0], "p_out")
    points = tuple(fm.SelectionPoint(f"distribution.{owner}", "distribution", (fm.Candidate("old"),),
                                    default_candidate="old", owner=owner) for owner in owners)
    records = tuple(fm.SelectionRecord(point.id, "old", origin="test", policy="test") for point in points)
    source = fm.verify_module(replace(source, selection_points=points, selections=records))
    result = specialize_persistent_state_abi(source, pairs)
    assert result.selection_points == (points[1],)
    assert result.selections == (records[1],)


def test_selected_input_cannot_feed_two_operands_of_the_same_call():
    source, pairs = state_module()
    call = source.node_map["call0"]
    source = fm.verify_module(replace(source, nodes=tuple(
        replace(node, inputs=(call.inputs[0], call.inputs[0])) if node.id == call.id else node
        for node in source.nodes)))
    with pytest.raises(IRVerificationError, match="exactly one operand"):
        specialize_persistent_state_abi(source, pairs)


def test_materialized_distribution_keeps_the_incoming_high_level_dialect():
    source, pairs = state_module()
    source = fm.verify_module(replace(source, dialect="high_level", stage="distribution_candidates"))
    result = specialize_persistent_state_abi(source, pairs)
    assert all(isinstance(result.node_map[parameter].type, fm.DistributedType)
               for parameter in result.function_map[result.entry].parameters)
