# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig

from .helpers import packed_matmul_module


def test_pass_uses_ortools_and_materializes_canonical_program_output_view(tmp_path):
    module = packed_matmul_module()
    result = AutoDistributedPass.run(module, NvidiaSm90Target())

    selected = result.selection_map["distribution.output"]
    distributed_compute = result.node_map["output"]
    function_output = result.function_map["main"].outputs[0]

    assert selected.origin == "ortools-cp-sat"
    selected_candidate = next(
        candidate
        for point in result.selection_points
        if point.id == "distribution.output"
        for candidate in point.candidates
        if candidate.id == selected.candidate_id
    )
    assert "matmul-" in selected_candidate.parameters["reason"]
    assert ".in_" in selected.candidate_id
    assert ".out_" in selected.candidate_id
    assert isinstance(distributed_compute.type, fm.DistributedType)
    assert any(node.op == "distributed.sharded_view" for node in result.nodes)
    assert all(
        node.metadata["realization"] == node.op.removeprefix("distributed.")
        for node in result.nodes if node.op.startswith("distributed.")
    )
    assert result.node_map[function_output].op == "distributed.sharded_view"
    assert isinstance(result.node_map[function_output].type, fm.DistributedType)
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in result.node_map[function_output].type.axis_policies
    )
    assert result.metadata["auto_distribution"]["reshard_nodes"] >= 3

    path = fm.emit_module(result, tmp_path / "distributed.py")
    assert fm.load_module(path) == result


def test_proposal_is_editable_before_any_distributed_node_is_materialized():
    module = packed_matmul_module()
    proposed = AutoDistributedPass.propose(module, NvidiaSm90Target())

    assert proposed.nodes == module.nodes
    assert proposed.functions == module.functions
    assert not any(
        node.op.startswith("distributed.") for node in proposed.nodes
    )
    point = next(
        point
        for point in proposed.selection_points
        if point.id == "distribution.output"
    )
    assert len(point.candidates) > 1
    assert proposed.selection_map[point.id].candidate_id == point.default_candidate
    assert proposed.metadata["auto_distribution_proposal"]["solver"] == (
        "ortools-cp-sat"
    )


def test_apply_preserves_agent_selection_provenance_after_materialization():
    target = NvidiaSm90Target()
    proposed = AutoDistributedPass.propose(packed_matmul_module(), target)
    point = next(
        point
        for point in proposed.selection_points
        if point.id == "distribution.output"
    )
    records = tuple(
        replace(
            record,
            origin="agent",
            policy="agent-override/v1",
            rationale="Keep the measured distribution.",
        )
        if record.point_id == point.id
        else record
        for record in proposed.selections
    )

    result = AutoDistributedPass.apply(
        replace(proposed, selections=records), target
    )

    selected = result.selection_map[point.id]
    assert selected.origin == "agent"
    assert selected.policy == "agent-override/v1"
    assert selected.rationale == "Keep the measured distribution."
    assert "auto_distribution_proposal" not in result.metadata
    assert any(node.op.startswith("distributed.") for node in result.nodes)


def test_pass_rejects_multiple_unselected_topology_alternatives():
    class MultiplePlacementTarget:
        name = "multiple-placement-test"
        distribution_policy = type(
            "Policy", (), {"identity": "multiple-placement-test/v1"})()

        def distributed_placements(self, module):
            del module
            return (
                fm.Placement((8, 16), "yx", "bb"),
                fm.Placement((4, 32), "yx", "bb"),
            )

    with pytest.raises(ValueError, match="exactly one selected Placement"):
        AutoDistributedPass.run(
            packed_matmul_module(), MultiplePlacementTarget())


def test_distribution_selection_cannot_diverge_from_materialized_ir():
    result = AutoDistributedPass.run(packed_matmul_module(), NvidiaSm90Target())
    point = next(point for point in result.selection_points if point.id == "distribution.output")
    selected = result.selection_map[point.id]
    alternative = next(candidate.id for candidate in point.candidates if candidate.id != selected.candidate_id)
    selections = tuple(
        replace(record, candidate_id=alternative) if record.point_id == point.id else record
        for record in result.selections
    )

    with pytest.raises(IRVerificationError, match="does not match the materialized candidate"):
        fm.verify_module(replace(result, selections=selections))


def test_reusable_function_output_and_call_share_one_distributed_abi():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("float32", (16,))
    layer_arg = builder.var("layer_arg", value_type, id="layer_arg")
    layer_result = builder.call(
        "math.silu", (layer_arg,), value_type, id="layer_result"
    )
    builder.function("layer", (layer_arg,), (layer_result,))
    main_arg = builder.var("main_arg", value_type, id="main_arg")
    main_result = builder.call(
        "builtin.call",
        (main_arg,),
        value_type,
        id="main_result",
        attrs={"callee": "layer"},
    )
    builder.function("main", (main_arg,), (main_result,))

    result = AutoDistributedPass.run(builder.build(entry="main"), NvidiaSm90Target())

    call_type = result.node_map["main_result"].type
    layer_output = result.node_map[result.function_map["layer"].outputs[0]]
    assert isinstance(call_type, fm.DistributedType)
    assert layer_output.type == call_type
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in call_type.axis_policies
    )
    assert all(
        result.node_map[node.inputs[0]].type != node.type
        for node in result.nodes
        if node.op in {"distributed.boxing", "distributed.sharded_view"}
    )


def test_reusable_function_attribute_scalars_stay_in_the_immediate_abi():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    slots_type = fm.tensor_type("bfloat16", (1, 1, 8))
    state_type = PagedAttentionStateConfig(
        1, 1, 8, block_size=4, num_blocks=2, lanes=1
    ).ref_type
    layer_type = fm.tensor_type("int32", ())
    advance_type = fm.tensor_type("bool", ())

    slots = builder.var("slots", slots_type, id="decode_slots")
    state = builder.var("state", state_type, id="decode_state")
    layer = builder.var("layer_id", layer_type, id="decode_layer_id")
    advance = builder.var(
        "advance_sequence", advance_type, id="decode_advance_sequence"
    )
    update = builder.call(
        "nn.update_paged_attention_kv_cache",
        (slots, state, layer, advance),
        state_type,
        id="decode_update",
        effect=fm.effect("read_write", "paged_attention_kv_cache"),
        attrs={"cache_kind": "value", "layout": ("seq", "head", "dim")},
    )
    builder.function("decode", (slots, state, layer, advance), (update,))

    main_slots = builder.var("main_slots", slots_type, id="main_slots")
    main_state = builder.var("main_state", state_type, id="main_state")
    main_layer = builder.node(
        op="builtin.scalar_const", type=layer_type, id="main_layer",
        attrs={"value": 3},
    )
    main_advance = builder.node(
        op="builtin.scalar_const", type=advance_type, id="main_advance",
        attrs={"value": True},
    )
    call = builder.call(
        "builtin.call",
        (main_slots, main_state, main_layer, main_advance),
        state_type,
        id="main_call",
        effect=fm.effect("read_write", "paged_attention_kv_cache"),
        attrs={"callee": "decode"},
    )
    builder.function("main", (main_slots, main_state), (call,))

    result = AutoDistributedPass.run(builder.build(entry="main"), NvidiaSm90Target())

    call = result.node_map["main_call"]
    assert call.inputs[2:] == ("main_layer", "main_advance")
    assert result.node_map["decode_layer_id"].type == layer_type
    assert result.node_map["decode_advance_sequence"].type == advance_type
    assert result.node_map["decode_update"].inputs[2:] == (
        "decode_layer_id", "decode_advance_sequence"
    )
    assert not any(
        node.op == "distributed.boxing"
        and node.inputs[0] in {
            "main_layer", "main_advance", "decode_layer_id",
            "decode_advance_sequence",
        }
        for node in result.nodes
    )
