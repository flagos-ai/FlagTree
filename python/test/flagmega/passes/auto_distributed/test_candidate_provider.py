# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.providers import (
    GdnConvolutionCandidateProvider,
    GdnRecurrentCandidateProvider,
    MatMulCandidateProvider,
)
from triton.flagmega.passes.auto_distributed import providers
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

from .helpers import packed_matmul_module


def test_matmul_provider_returns_replicated_output_split_and_partial_k_split():
    module = packed_matmul_module()
    node = module.node_map["output"]
    placement = fm.Placement((8,), "b", "b")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
    )

    provider = MatMulCandidateProvider()
    return_types = provider.get_return_candidate_types(context, (node.type,))
    candidates = provider.get_candidates(context)
    by_reason = {candidate.reason: candidate for candidate in candidates}

    assert return_types == tuple(candidate.return_type for candidate in candidates)
    assert provider.create_candidate_target(context, return_types[0]) == node.op
    for candidate in candidates:
        tuples = provider.try_get_input_type_tuples(context, candidate.return_type)
        assert tuples is not None
        assert any(value.input_types == candidate.input_types for value in tuples)
    assert set(by_reason) == {"broadcast-replicated", "matmul-output-sbp", "matmul-K-sbp-partial"}
    replicated = by_reason["broadcast-replicated"]
    assert isinstance(replicated.return_type, fm.DistributedType)
    assert all(isinstance(value, fm.DistributedType) for value in replicated.input_types)
    assert all(isinstance(policy, fm.SBPBroadCast) for policy in replicated.return_type.axis_policies)
    assert isinstance(by_reason["matmul-output-sbp"].return_type, fm.DistributedType)
    assert isinstance(by_reason["matmul-K-sbp-partial"].return_type.partial, fm.SBPPartial)
    assert by_reason["matmul-output-sbp"].input_types[1].axis_policies[0].hierarchy_axes == (0,)


def test_matmul_provider_uses_exact_two_dimensional_mesh_axes_and_divisor():
    module = packed_matmul_module(size=4096)
    node = module.node_map["output"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
    )

    candidates = MatMulCandidateProvider().get_candidates(context)
    full_mesh_output = next(
        candidate
        for candidate in candidates
        if candidate.reason == "matmul-output-sbp"
        and candidate.return_type.axis_policies[-1].hierarchy_axes == (0, 1)
    )
    reduction_axes = {
        candidate.return_type.partial.axes
        for candidate in candidates
        if candidate.reason == "matmul-K-sbp-partial"
    }
    assert (
        full_mesh_output.return_type.tensor.shape[-1].fixed_value
        // full_mesh_output.return_type.placement.size
        == 32
    )
    assert full_mesh_output.input_types[1].axis_policies[0].hierarchy_axes == (0, 1)
    # Block-FP8 K shards must end on a 128-value scale-group boundary.
    # Individual mesh axes leave K=512/K=256 per owner; combining them would
    # leave K=32 and is therefore not a legal reduction candidate.
    assert reduction_axes == {(0,), (1,)}


def test_dense_matmul_provider_offers_disjoint_2d_output_and_reduction_split():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value = builder.var(
        "value", fm.tensor_type("bfloat16", [1, 4096]), id="value"
    )
    weight = builder.weight(
        "weight",
        fm.tensor_type("bfloat16", [4096, 4096]),
        source="unit.safetensors",
        key="weight",
        id="weight",
    )
    output = builder.call(
        "math.matmul",
        (value, weight),
        fm.tensor_type("bfloat16", [1, 4096]),
        id="output",
        attrs={"transpose_a": False, "transpose_b": False},
    )
    builder.function("main", (value,), (output,))
    module = fm.verify_module(builder.build(entry="main"))
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        output,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in output.inputs),
    )

    candidate = next(
        candidate
        for candidate in MatMulCandidateProvider().get_candidates(context)
        if candidate.reason == "matmul-output-K-sbp-partial"
        and candidate.return_type.axis_policies[-1].hierarchy_axes == (1,)
        and candidate.return_type.partial == fm.SBP.partial((0,))
    )

    assert candidate.input_types[0].axis_policies[1].hierarchy_axes == (0,)
    assert candidate.input_types[1].axis_policies[0].hierarchy_axes == (0,)
    assert candidate.input_types[1].axis_policies[1].hierarchy_axes == (1,)


def test_matmul_provider_does_not_confuse_one_mesh_axis_with_the_full_mesh():
    module = packed_matmul_module(size=160)
    node = module.node_map["output"]
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
    )

    candidates = MatMulCandidateProvider().get_candidates(context)
    output_axes = {
        candidate.return_type.axis_policies[-1].hierarchy_axes
        for candidate in candidates
        if candidate.reason == "matmul-output-sbp"
    }
    reduction_axes = {
        candidate.return_type.partial.axes
        for candidate in candidates
        if candidate.reason == "matmul-K-sbp-partial"
    }

    # Logical N=160 can be split by y=8 and x=16 independently, but not by
    # the combined 8*16 mesh.  Packed physical K=5 is not splittable.
    assert output_axes == {(0,), (1,)}
    assert reduction_axes == set()


def test_provider_does_not_offer_illegal_k_split_for_packed_physical_extent():
    module = packed_matmul_module(size=128)
    node = module.node_map["output"]
    context = DistributedCandidateContext(
        module,
        node,
        fm.Placement((8,), "b", "b"),
        tuple((module.node_map[input_id].type,) for input_id in node.inputs),
    )

    reasons = {candidate.reason for candidate in MatMulCandidateProvider().get_candidates(context)}

    assert "matmul-output-sbp" in reasons
    assert "matmul-K-sbp-partial" not in reasons


def test_dense_matmul_provider_uses_original_rhs_axes_for_transpose_b():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value = builder.var("value", fm.tensor_type("bfloat16", [1, 6144]), id="value")
    weight = builder.weight(
        "weight", fm.tensor_type("bfloat16", [2048, 6144]),
        source="unit.safetensors", key="weight", id="weight")
    output = builder.call(
        "math.matmul", (value, weight), fm.tensor_type("bfloat16", [1, 2048]), id="output",
        attrs={"transpose_a": False, "transpose_b": True})
    builder.function("main", (value,), (output,))
    module = fm.verify_module(builder.build(entry="main"))
    context = DistributedCandidateContext(
        module, output, fm.Placement((8,), "b", "b"),
        tuple((module.node_map[input_id].type,) for input_id in output.inputs))

    candidates = {candidate.reason: candidate for candidate in MatMulCandidateProvider().get_candidates(context)}

    output_split = candidates["matmul-output-sbp"]
    reduction_split = candidates["matmul-K-sbp-partial"]
    assert output_split.input_types[1].axis_policies[0].hierarchy_axes == (0,)
    assert isinstance(output_split.input_types[1].axis_policies[1], fm.SBPBroadCast)
    assert reduction_split.input_types[1].axis_policies[1].hierarchy_axes == (0,)
    assert isinstance(reduction_split.input_types[1].axis_policies[0], fm.SBPBroadCast)


def test_packed_dense_block_cyclic_unit_comes_from_injected_target_policy():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value = builder.var(
        "value", fm.tensor_type("bfloat16", [1, 2048]), id="value"
    )
    weight = builder.weight(
        "weight",
        fm.tensor_type("bfloat16", [128, 18992, 2, 64]),
        source="unit.safetensors",
        key="weight",
        id="weight",
    )
    output = builder.call(
        "math.packed_dense_matmul",
        (value, weight),
        fm.tensor_type("bfloat16", [1, 151936]),
        id="output",
        attrs={"packed_layout": "k_major_n8_k16", "logical_n": None},
    )
    builder.function("main", (value,), (output,))
    module = fm.verify_module(builder.build(entry="main"))
    placement = fm.Placement((8, 16), "yx", "bb")
    context = DistributedCandidateContext(
        module,
        output,
        placement,
        tuple((module.node_map[input_id].type,) for input_id in output.inputs),
        PyNttDistributedSplitCandidateProvider(block_bytes=128),
    )

    candidate = next(
        candidate
        for candidate in MatMulCandidateProvider().get_candidates(context)
        if candidate.reason == "matmul-output-sbp"
        and candidate.return_type.axis_policies[1]
        == fm.SBP.split_block_cyclic((0, 1), 64)
    )
    weight_policy = candidate.input_types[1].axis_policies[1]
    output_policy = candidate.return_type.axis_policies[1]

    assert weight_policy.hierarchy_axes == (0, 1)
    assert weight_policy.stages[0].distribution.block_size == 8
    assert output_policy.hierarchy_axes == (0, 1)
    assert output_policy.stages[0].distribution.block_size == 64


def test_generic_candidate_providers_do_not_construct_physical_block_policy():
    source = inspect.getsource(providers)

    assert "split_block_cyclic" not in source
    assert "is_physical_block_axis" not in source
    assert "sm90" not in source.lower()
    assert "nvidia" not in source.lower()


def test_gdn_convolution_provider_splits_projected_channels_over_full_mesh():
    placement = fm.Placement((2, 4), "xy", "bb")
    state = fm.RefType("state", (("convolution", fm.tensor_type("bfloat16", [1, 10240, 3])),))
    inputs = (
        fm.tensor_type("bfloat16", [1, 10240]),
        state,
        fm.tensor_type("bfloat16", [10240, 1, 4]),
    )
    fields = (inputs[0], state)
    module, call = _provider_module("nn.gdn_convolution", inputs, fm.TupleType(fields))

    candidates = GdnConvolutionCandidateProvider().get_candidates(
        DistributedCandidateContext(module, call, placement, tuple((value,) for value in inputs)))
    channel_split = next(
        candidate for candidate in candidates
        if candidate.reason == "gdn-convolution-channel-sbp")

    assert isinstance(channel_split.return_type, fm.TupleType)
    assert channel_split.return_type.fields[0].axis_policies[1].hierarchy_axes == (0, 1)
    assert channel_split.return_type.fields[1] == state
    assert channel_split.input_types[0].axis_policies[1].hierarchy_axes == (0, 1)
    assert channel_split.input_types[2].axis_policies[0].hierarchy_axes == (0, 1)
    assert all(
        isinstance(policy, fm.SBPBroadCast)
        for policy in channel_split.input_types[2].axis_policies[1:]
    )


def test_gdn_recurrent_provider_returns_materialized_value_head_split():
    placement = fm.Placement((2, 4), "xy", "bb")
    state = fm.RefType("state", (("recurrent", fm.tensor_type("float32", [1, 48, 128, 128])),))
    inputs = (
        state,
        fm.tensor_type("bfloat16", [1, 10240]),
        fm.tensor_type("bfloat16", [1, 6144]),
        fm.tensor_type("bfloat16", [1, 5120]),
        fm.tensor_type("bfloat16", [48, 5120]),
        fm.tensor_type("bfloat16", [48, 5120]),
        fm.tensor_type("float32", [48]),
        fm.tensor_type("float32", [48]),
        fm.tensor_type("bfloat16", [128]),
    )
    output = fm.TupleType((fm.tensor_type("bfloat16", [1, 6144]), state))
    module, call = _provider_module("nn.gdn_recurrent_core", inputs, output)

    candidates = GdnRecurrentCandidateProvider().get_candidates(
        DistributedCandidateContext(module, call, placement, tuple((value,) for value in inputs)))
    head_split = next(candidate for candidate in candidates if candidate.reason == "gdn-recurrent-head-sbp")
    hidden = head_split.return_type.fields[0]

    assert isinstance(hidden, fm.DistributedType)
    assert hidden.partial is None
    assert hidden.axis_policies[1].hierarchy_axes == (0, 1)
    assert head_split.input_types[2].axis_policies[1].hierarchy_axes == (0, 1)
    assert all(
        all(isinstance(policy, fm.SBPBroadCast) for policy in value.axis_policies)
        for index, value in enumerate(head_split.input_types)
        if index not in (0, 2)
    )


def test_gdn_stage_providers_reuse_legal_upstream_sum_partial_inputs():
    placement = fm.Placement((2, 4), "xy", "bb")
    state = fm.RefType("state")
    qkv = fm.tensor_type("bfloat16", [1, 10240])
    conv_weight = fm.tensor_type("bfloat16", [10240, 1, 4])
    module, call = _provider_module(
        "nn.gdn_convolution", (qkv, state, conv_weight), fm.TupleType((qkv, state)))
    partial_qkv = fm.DistributedType(
        qkv,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
        partial=fm.SBP.partial((1,)),
    )
    available = ((partial_qkv,), (state,), (conv_weight,))

    candidates = GdnConvolutionCandidateProvider().get_candidates(
        DistributedCandidateContext(module, call, placement, available))

    direct = next(
        candidate for candidate in candidates
        if candidate.reason == "gdn-convolution-direct-sum-partial")
    assert direct.input_types[0] == partial_qkv
    assert direct.return_type.fields[0].partial is None
    assert direct.return_type.fields[0].axis_policies[1].hierarchy_axes == (0, 1)


def _provider_module(op: str, input_types, output_type):
    leaves = tuple(
        fm.Node(f"input_{index}", "builtin.var", (), value, attrs={"name": f"input_{index}"})
        for index, value in enumerate(input_types)
    )
    call = fm.Node("call", op, tuple(node.id for node in leaves), output_type)
    module = fm.IRModule(
        dialect="high_level",
        stage="packed",
        nodes=(*leaves, call),
        functions=(fm.Function("main", (), (call.id,)),),
        entry="main",
    )
    return module, call
