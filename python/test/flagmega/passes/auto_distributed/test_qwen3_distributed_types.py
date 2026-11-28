# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.passes.auto_distributed import AutoDistributedPass
from triton.flagmega.stages import get_stage
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import qwen3_packed_module


@pytest.fixture(scope="module")
def distributed_module():
    target = NvidiaSm90Target()
    return target.auto_distribute(_run_post_packing(qwen3_packed_module(), target))


def _run_post_packing(module, target):
    for stage_name in (
            "propagate-function-boundary-layouts",
            "post-function-boundary-pack-propagation",
            "thread-norm-stats",
            "decompose-paged-attention",
            "form-add-norm-stats",
    ):
        module = get_stage(stage_name).run(module, target)
    return module


def test_qwen3_h800_uses_reference_two_dimensional_block_mesh(distributed_module):
    result = distributed_module
    placement = fm.Placement((8, 16), "yx", "bb")
    distributed = [leaf for node in result.nodes for leaf in _distributed_tensor_leaves(node.type)]
    assert distributed
    assert all(leaf.placement == placement for leaf in distributed)
    gate_up = result.node_map["mlp_gate_up"].type
    assert isinstance(gate_up, fm.DistributedType)
    assert gate_up.axis_policies[-1].hierarchy_axes == (0, 1)
    down_node = result.node_map["mlp_down.vectorized.compute"]
    down = down_node.type
    assert isinstance(down, fm.DistributedType)
    # Joint epilogue/publication costs can favor output-N owners over an
    # all-K split. Check the distribution contract, not one optimizer pick.
    split_axes = {
        axis
        for policy in down.axis_policies if isinstance(policy, fm.SBPSplit) for axis in policy.hierarchy_axes
    }
    partial_axes = set(down.partial.axes) if down.partial is not None else set()
    assert split_axes.isdisjoint(partial_axes)
    assert split_axes | partial_axes == {0, 1}
    assert fm.get_definition(down_node.op).infer_type(tuple(result.node_map[value] for value in down_node.inputs),
                                                      down_node.attrs) == down


def test_post_attention_norm_replication_avoids_internal_widening_view():

    class Norm(fm.Module):

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 128)))
            weight = self.input("weight", fm.tensor_type("bfloat16", (128, )))
            result = fm.F.nn.rms_norm(value, weight, epsilon=1e-6, name="post_attention_norm")
            self.function("main", (value, weight), (result, ))

    module = Norm(dialect="high_level", stage="imported", entry="main").build()
    compiler = Compiler(CompileOptions(placement=fm.Placement((2, 2), "yx", "bb")))
    proposed = compiler.compile(module, stop_after="propose-distribution").module
    points = tuple(point for point in proposed.selection_points if point.kind == "distribution" if point.owner in {
        "post_attention_norm.vectorized.compute",
        "post_attention_norm",
    })
    assert {point.owner
            for point in points} == {
                "post_attention_norm.vectorized.compute",
                "post_attention_norm",
            }
    broadcast_ids = {
        point.id:
        next(candidate.id
             for candidate in point.candidates
             if str(candidate.parameters["return_type"]).endswith(";B,B)"))
        for point in points
    }
    selections = tuple(
        replace(
            record,
            candidate_id=broadcast_ids[record.point_id],
            origin="agent",
            policy="agent-no-widening-view/v1",
            rationale="Keep post-attention norm block-replicated.",
        ) if record.point_id in broadcast_ids else record for record in proposed.selections)
    result = AutoDistributedPass.apply(replace(proposed, selections=selections), compiler.target)

    normalized = result.node_map["post_attention_norm.vectorized.compute"]
    assert isinstance(normalized.type, fm.DistributedType)
    assert all(isinstance(policy, fm.SBPBroadCast) for policy in normalized.type.axis_policies)
    assert not any(node.op == "distributed.sharded_view" and node.inputs == ("post_attention_norm", )
                   for node in result.nodes)


def test_qwen3_every_compute_tensor_leaf_is_distributed_after_auto_dist(distributed_module):
    result = distributed_module

    for node in result.nodes:
        if not node.inputs:
            # Function parameters and checkpoint storage are originators.  As
            # in reference, an explicit ShardedView introduces their dist type.
            continue
        assert not _logical_tensor_leaves(
            node.type), (f"{node.id} ({node.op}) retained logical tensor result {node.type!r}")
        if node.op.startswith("distributed."):
            continue
        definition = fm.get_definition(node.op)
        for input_id, parameter in zip(node.inputs, definition.input_parameters):
            # Match the reference VisitLeafArgument behavior: Attribute operands
            # remain editable graph values but have no SBP/distributed type.
            if parameter.parameter_kind != fm.ParameterKind.INPUT:
                continue
            input_type = result.node_map[input_id].type
            assert not _logical_tensor_leaves(input_type), (f"{node.id} ({node.op}) consumes logical tensor input "
                                                            f"{input_id}: {input_type!r}")


def test_qwen3_attention_tuple_and_program_output_keep_distributed_types(tmp_path, distributed_module):
    result = distributed_module

    # AutoPacking's reference-aligned FoldGetItemTuple removes the temporary
    # wrapper tuple.  The packed projection is the surviving semantic
    # multi-result producer and selection owner.
    qkv_type = result.node_map["self_attention_qkv_projection.packed_projection"].type
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
    vectorized_qkv = result.node_map["updated_state.qkv_rope_with_cache.vectorized.qkv"].type
    assert isinstance(vectorized_qkv, fm.TupleType)
    assert all(isinstance(field, fm.DistributedType) for field in vectorized_qkv.fields)
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


def test_qwen3_decomposed_attention_preserves_qkv_output_sharding(distributed_module):
    result = distributed_module

    projection = result.node_map["self_attention_qkv_projection.packed_projection"].type
    assert isinstance(projection, fm.TupleType)
    assert all(
        isinstance(field, fm.DistributedType) and isinstance(field.axis_policies[-1], fm.SBPSplit)
        and field.partial is None for field in projection.fields)
    assert tuple(field.axis_policies[-1] for field in projection.fields) == (
        fm.SBP.split_block_cyclic((0, 1), 2),
        fm.SBP.split_block_cyclic((0, 1), 1),
        fm.SBP.split_block_cyclic((0, 1), 1),
    )
    fused = result.node_map["updated_state.qkv_rope_with_cache"]
    assert len(fused.inputs) == 12
    for stats_id in fused.inputs[10:12]:
        stats_type = result.node_map[stats_id].type
        assert isinstance(stats_type, fm.DistributedType)
        assert stats_type.partial is None
    qkv_input = result.node_map["updated_state.qkv_rope_with_cache.vectorized.qkv"].type
    assert isinstance(qkv_input, fm.TupleType)
    assert all(
        isinstance(field, fm.DistributedType) and isinstance(field.axis_policies[1], fm.SBPSplit)
        for field in qkv_input.fields)
    assert qkv_input.fields[1].axis_policies[1] == qkv_input.fields[2].axis_policies[1]
    assert "self_attention_query_rope" not in result.node_map
    assert "self_attention_key_rope" not in result.node_map
    fused_type = result.node_map["updated_state.qkv_rope_with_cache"].type
    assert isinstance(fused_type, fm.TupleType)
    query_type = fused_type.fields[0]
    assert isinstance(query_type, fm.DistributedType)
    # Attention keeps head sharding and reduces only over the context axes.
    # The solver can assign those roles to different physical axes.
    partial = result.node_map["self_attention_paged_attention.partial"]
    partial_query = result.node_map[partial.inputs[0]].type
    assert isinstance(partial_query, fm.DistributedType)
    sequence_policy, head_policy, dimension_policy = partial_query.axis_policies
    assert sequence_policy == dimension_policy == fm.SBP.broadcast()
    assert isinstance(head_policy, fm.SBPSplit)
    attention_type = result.node_map["self_attention_paged_attention"].type
    assert isinstance(attention_type, fm.DistributedType)
    assert attention_type.axis_policies == partial_query.axis_policies
    assert attention_type.partial is None
    partial_type = result.node_map["self_attention_paged_attention.partial"].type
    assert isinstance(partial_type, fm.TupleType)
    assert all(field.axis_policies[1] == head_policy for field in partial_type.fields)
    assert tuple(field.partial.reduce_op for field in partial_type.fields) == (
        fm.ReduceOp.MAX,
        fm.ReduceOp.SUM,
        fm.ReduceOp.SUM,
    )
    context_axes = set(range(partial_query.placement.rank)) - set(head_policy.hierarchy_axes)
    assert context_axes
    assert all(set(field.partial.axes) == context_axes for field in partial_type.fields)
    assert not any(node.op == "distributed.sharded_view" and node.inputs == ("self_attention_paged_attention", )
                   for node in result.nodes)


def test_qwen3_originators_keep_logical_storage_with_distributed_views(distributed_module):
    result = distributed_module

    assert isinstance(result.node_map["input_ids"].type, fm.TensorType)
    assert isinstance(result.node_map["w_embed_tokens_weight"].type, fm.TensorType)
    for source in ("input_ids", "w_embed_tokens_weight"):
        views = [node for node in result.nodes if node.op == "distributed.sharded_view" and node.inputs == (source, )]
        assert views
        assert all(
            isinstance(node.type, fm.DistributedType) and node.type.tensor == result.node_map[source].type
            for node in views)


def _logical_tensor_leaves(value: fm.IRType) -> tuple[fm.TensorType, ...]:
    if isinstance(value, fm.DistributedType):
        return ()
    if isinstance(value, fm.TensorType):
        return (value, )
    if isinstance(value, fm.TupleType):
        return tuple(leaf for field in value.fields for leaf in _logical_tensor_leaves(field))
    # Reference fields describe mutable state storage, not distributed value
    # leaves in the expression graph.
    return ()


def _distributed_tensor_leaves(value: fm.IRType) -> tuple[fm.DistributedType, ...]:
    if isinstance(value, fm.DistributedType):
        return (value, )
    if isinstance(value, fm.TupleType):
        return tuple(leaf for field in value.fields for leaf in _distributed_tensor_leaves(field))
    return ()
