# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.stages import get_stage
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import qwen3_packed_module


def _auto_distribute():
    target = NvidiaSm90Target()
    module = _run_post_packing(qwen3_packed_module(), target)
    return target.auto_distribute(module)


def _run_post_packing(module, target):
    for stage_name in (
        "propagate-function-boundary-layouts",
        "post-function-boundary-pack-propagation",
        "thread-norm-stats",
        "decompose-paged-attention",
        "form-matmul-norm-stats-combine",
    ):
        module = get_stage(stage_name).run(module, target)
    return module


def test_qwen3_h800_uses_nncase_two_dimensional_block_mesh():
    module = qwen3_packed_module()
    target = NvidiaSm90Target()

    assert target.distributed_placements(module) == (
        fm.Placement((8, 16), "yx", "bb"),
    )

    result = target.auto_distribute(_run_post_packing(module, target))
    placement = fm.Placement((8, 16), "yx", "bb")
    distributed = [
        leaf
        for node in result.nodes
        for leaf in _distributed_tensor_leaves(node.type)
    ]
    assert distributed
    assert all(leaf.placement == placement for leaf in distributed)
    gate_up = result.node_map["mlp_gate_up"].type
    assert isinstance(gate_up, fm.DistributedType)
    assert gate_up.axis_policies[-1].hierarchy_axes == (0, 1)
    down = result.node_map["mlp_down.vectorized.compute"].type
    assert isinstance(down, fm.DistributedType)
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in down.axis_policies
    )
    assert down.partial == fm.SBP.partial((0, 1))


def test_post_attention_norm_replication_avoids_internal_widening_view():
    target = NvidiaSm90Target()
    module = _run_post_packing(qwen3_packed_module(), target)
    proposed = AutoDistributedPass.propose(module, target)
    points = tuple(
        point
        for point in proposed.selection_points
        if point.kind == "distribution"
        if point.owner in {
            "post_attention_norm.vectorized.compute",
            "post_attention_norm",
        }
    )
    assert {point.owner for point in points} == {
        "post_attention_norm.vectorized.compute",
        "post_attention_norm",
    }
    broadcast_ids = {
        point.id: next(
            candidate.id
            for candidate in point.candidates
            if str(candidate.parameters["return_type"]).endswith(";B,B)")
        )
        for point in points
    }
    selections = tuple(
        replace(
            record,
            candidate_id=broadcast_ids[record.point_id],
            origin="agent",
            policy="agent-no-widening-view/v1",
            rationale="Keep post-attention norm block-replicated.",
        )
        if record.point_id in broadcast_ids
        else record
        for record in proposed.selections
    )
    result = AutoDistributedPass.apply(
        replace(proposed, selections=selections), target
    )

    normalized = result.node_map["post_attention_norm.vectorized.compute"]
    assert isinstance(normalized.type, fm.DistributedType)
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in normalized.type.axis_policies
    )
    assert not any(
        node.op == "distributed.sharded_view"
        and node.inputs == ("post_attention_norm",)
        for node in result.nodes
    )


def test_qwen3_every_compute_tensor_leaf_is_distributed_after_auto_dist():
    result = _auto_distribute()

    for node in result.nodes:
        if not node.inputs:
            # Function parameters and checkpoint storage are originators.  As
            # in nncase, an explicit ShardedView introduces their dist type.
            continue
        assert not _logical_tensor_leaves(node.type), (
            f"{node.id} ({node.op}) retained logical tensor result {node.type!r}"
        )
        if node.op.startswith("distributed."):
            continue
        definition = fm.get_definition(node.op)
        for input_id, parameter in zip(node.inputs, definition.input_parameters):
            # Match nncase's VisitLeafArgument behavior: Attribute operands
            # remain editable graph values but have no SBP/distributed type.
            if parameter.parameter_kind != fm.ParameterKind.INPUT:
                continue
            input_type = result.node_map[input_id].type
            assert not _logical_tensor_leaves(input_type), (
                f"{node.id} ({node.op}) consumes logical tensor input "
                f"{input_id}: {input_type!r}"
            )


def test_qwen3_attention_tuple_and_program_output_keep_distributed_types(tmp_path):
    result = _auto_distribute()

    # AutoPacking's nncase-aligned FoldGetItemTuple removes the temporary
    # wrapper tuple.  The packed projection is the surviving semantic
    # multi-result producer and selection owner.
    qkv_type = result.node_map[
        "self_attention_qkv_projection.packed_projection"
    ].type
    assert isinstance(qkv_type, fm.TupleType)
    assert all(isinstance(field, fm.DistributedType) for field in qkv_type.fields)
    rotary, = (node for node in result.nodes if node.op == "nn.rotary_embedding")
    rotary_type = rotary.type
    assert isinstance(rotary_type, fm.TupleType)
    assert all(isinstance(field, fm.DistributedType) for field in rotary_type.fields)
    assert all(isinstance(field.tensor.dtype, fm.VectorType) for field in rotary_type.fields)
    fused = result.node_map["updated_state.qkv_rope_with_cache"]
    assert isinstance(fused.type, fm.TupleType)
    assert isinstance(fused.type.fields[0], fm.DistributedType)
    assert isinstance(fused.type.fields[1], fm.RefType)
    vectorized_qkv = result.node_map[
        "updated_state.qkv_rope_with_cache.vectorized.qkv"
    ].type
    assert isinstance(vectorized_qkv, fm.TupleType)
    assert all(
        isinstance(field, fm.DistributedType)
        for field in vectorized_qkv.fields
    )
    assert "updated_state.qkv" not in result.node_map
    assert "self_attention_key_cache_update" not in result.node_map
    assert "self_attention_value_cache_update" not in result.node_map
    assert isinstance(result.node_map["updated_state"].type, fm.RefType)
    assert isinstance(
        result.node_map["self_attention_paged_attention"].type,
        fm.DistributedType,
    )
    assert isinstance(
        result.node_map["attention_output.vectorized.compute"].type,
        fm.DistributedType,
    )

    output_id, state_id = result.function_map["main"].outputs
    output_type = result.node_map[output_id].type
    assert isinstance(output_type, fm.DistributedType)
    assert all(isinstance(policy, fm.SBPBroadCast) for policy in output_type.axis_policies)
    assert isinstance(result.node_map[state_id].type, fm.RefType)

    checkpoint = fm.emit_module(result, tmp_path / "auto_distributed.py")
    assert fm.load_module(checkpoint) == result


def test_qwen3_decomposed_attention_preserves_qkv_output_sharding():
    result = _auto_distribute()

    projection = result.node_map["self_attention_qkv_projection.packed_projection"].type
    assert isinstance(projection, fm.TupleType)
    assert all(
        isinstance(field, fm.DistributedType)
        and isinstance(field.axis_policies[-1], fm.SBPSplit)
        and field.partial is None
        for field in projection.fields
    )
    assert tuple(field.axis_policies[-1] for field in projection.fields) == (
        fm.SBP.split_block_cyclic((0, 1), 2),
        fm.SBP.split_block_cyclic((0, 1), 1),
        fm.SBP.split_block_cyclic((0, 1), 1),
    )
    for node_id in (
        "updated_state.qkv_rope_with_cache.vectorized.q",
        "updated_state.qkv_rope_with_cache.vectorized.k",
        "updated_state.qkv_rope_with_cache.vectorized.v",
    ):
        value_type = result.node_map[node_id].type
        assert isinstance(value_type, fm.DistributedType)
        assert all(
            isinstance(policy, fm.SBPBroadCast)
            for policy in value_type.axis_policies
        ), node_id
    qkv_input = result.node_map[
        "updated_state.qkv_rope_with_cache.vectorized.qkv"
    ].type
    assert isinstance(qkv_input, fm.TupleType)
    assert all(
        isinstance(field, fm.DistributedType)
        and isinstance(field.axis_policies[1], fm.SBPSplit)
        for field in qkv_input.fields
    )
    assert tuple(field.axis_policies[1].hierarchy_axes for field in qkv_input.fields) == (
        (1,),
        (0,),
        (0,),
    )
    assert "self_attention_query_rope" not in result.node_map
    assert "self_attention_key_rope" not in result.node_map
    fused_type = result.node_map["updated_state.qkv_rope_with_cache"].type
    assert isinstance(fused_type, fm.TupleType)
    query_type = fused_type.fields[0]
    assert isinstance(query_type, fm.DistributedType)
    # The fused producer may retain its canonical broadcast result, but the
    # paged-attention edge must expose the nncase head-local x view.
    partial = result.node_map["self_attention_paged_attention.partial"]
    partial_query = result.node_map[partial.inputs[0]].type
    assert isinstance(partial_query, fm.DistributedType)
    assert partial_query.axis_policies == (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), granularity=1),
        fm.SBP.broadcast(),
    )
    attention_type = result.node_map["self_attention_paged_attention"].type
    assert isinstance(attention_type, fm.DistributedType)
    assert attention_type.axis_policies == (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), granularity=1),
        fm.SBP.broadcast(),
    )
    partial_type = result.node_map["self_attention_paged_attention.partial"].type
    assert isinstance(partial_type, fm.TupleType)
    assert all(
        field.axis_policies[1] == fm.SBP.split_contiguous((1,), granularity=1)
        for field in partial_type.fields
    )
    assert tuple(field.partial.reduce_op for field in partial_type.fields) == (
        fm.ReduceOp.MAX,
        fm.ReduceOp.SUM,
        fm.ReduceOp.SUM,
    )
    assert all(field.partial.axes == (0,) for field in partial_type.fields)
    assert not any(
        node.op == "distributed.sharded_view"
        and node.inputs == ("self_attention_paged_attention",)
        for node in result.nodes
    )


def test_qwen3_originators_use_target_selected_copy_or_alias_realization():
    result = _auto_distribute()

    assert isinstance(result.node_map["input_ids"].type, fm.TensorType)
    assert isinstance(result.node_map["w_embed_tokens_weight"].type, fm.TensorType)
    input_boxing = [
        node for node in result.nodes
        if node.op == "distributed.boxing" and node.inputs == ("input_ids",)
    ]
    embedding_views = [
        node for node in result.nodes
        if node.op == "distributed.sharded_view" and node.inputs == ("w_embed_tokens_weight",)
    ]
    assert input_boxing
    assert embedding_views
    assert all(
        isinstance(node.type, fm.DistributedType)
        for node in (*input_boxing, *embedding_views)
    )


def _logical_tensor_leaves(value: fm.IRType) -> tuple[fm.TensorType, ...]:
    if isinstance(value, fm.DistributedType):
        return ()
    if isinstance(value, fm.TensorType):
        return (value,)
    if isinstance(value, fm.TupleType):
        return tuple(
            leaf
            for field in value.fields
            for leaf in _logical_tensor_leaves(field)
        )
    # Reference fields describe mutable state storage, not distributed value
    # leaves in the expression graph.
    return ()


def _distributed_tensor_leaves(value: fm.IRType) -> tuple[fm.DistributedType, ...]:
    if isinstance(value, fm.DistributedType):
        return (value,)
    if isinstance(value, fm.TupleType):
        return tuple(
            leaf
            for field in value.fields
            for leaf in _distributed_tensor_leaves(field)
        )
    return ()
