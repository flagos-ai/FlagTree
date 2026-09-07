# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionStateConfig,
)
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext,
    DistributedCandidateProviderRegistry,
    build_search_graph,
)
from triton.flagmega.passes.auto_distributed.qkv_rope_with_cache_provider import (
    QKVRoPEWithCacheCandidateProvider,
)
from triton.flagmega.targets import NvidiaSm90Target


PLACEMENT = fm.Placement((8, 16), "yx", "bb")


def test_provider_accepts_independent_q_and_kv_head_shards_on_a_2d_mesh():
    module, fused, inputs = _fused_module()
    q, k, v = inputs[0].type.fields
    q_dist = _head_split(q, (0,))
    k_dist = _head_split(k, (1,))
    v_dist = _head_split(v, (1,))
    qkv_dist = fm.TupleType((q_dist, k_dist, v_dist))
    broadcast_inputs = tuple(
        _broadcast(value.type)
        for value in inputs[1:7]
    )
    scalar_controls = tuple(value.type for value in inputs[8:10])

    candidates = QKVRoPEWithCacheCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            fused,
            PLACEMENT,
            (
                (qkv_dist,),
                *((value,) for value in broadcast_inputs),
                ((inputs[7].type),),
                *((value,) for value in scalar_controls),
            ),
        )
    )

    candidate = next(
        candidate
        for candidate in candidates
        if candidate.input_types[0] == qkv_dist
    )
    assert candidate.input_types[0] == qkv_dist
    assert isinstance(candidate.return_type, fm.TupleType)
    query = candidate.return_type.fields[0]
    assert isinstance(query, fm.DistributedType)
    assert query.axis_policies[1] == q_dist.axis_policies[1]
    assert candidate.return_type.fields[1] == inputs[7].type
    assert candidate.input_types[8:] == scalar_controls


def test_provider_terminates_distributed_scalar_attribute_candidates():
    module, fused, inputs = _fused_module()
    available = []
    for index, value in enumerate(inputs):
        if index == 0:
            available.append((fm.TupleType(tuple(_broadcast(field) for field in value.type.fields)),))
        elif index == 7:
            available.append((value.type,))
        else:
            available.append((_broadcast(value.type),))

    candidates = QKVRoPEWithCacheCandidateProvider().get_candidates(
        DistributedCandidateContext(module, fused, PLACEMENT, tuple(available))
    )

    assert candidates
    assert all(candidate.input_types[8] == inputs[8].type for candidate in candidates)
    assert all(candidate.input_types[9] == inputs[9].type for candidate in candidates)


def test_provider_rejects_a_rotary_dimension_split():
    module, fused, inputs = _fused_module()
    q, k, v = inputs[0].type.fields
    q_dim_split = fm.DistributedType(
        q,
        (
            fm.SBP.broadcast(),
            fm.SBP.broadcast(),
            fm.SBP.split_block_cyclic((0,), 16),
        ),
        PLACEMENT,
    )
    qkv_dist = fm.TupleType((q_dim_split, _broadcast(k), _broadcast(v)))
    available = []
    for index, value in enumerate(inputs):
        if index == 0:
            available.append((qkv_dist,))
        elif index == 7:
            available.append((value.type,))
        else:
            available.append((_broadcast(value.type),))

    candidates = QKVRoPEWithCacheCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module, fused, PLACEMENT, tuple(available)
        )
    )

    assert candidates
    assert all(candidate.input_types[0] != qkv_dist for candidate in candidates)


def test_provider_uses_the_registered_split_policy_for_head_local_qkv_layouts():
    module, fused, inputs = _fused_module()
    available = []
    for index, value in enumerate(inputs):
        if index == 0:
            available.append((
                fm.TupleType(tuple(_broadcast(field) for field in value.type.fields)),
            ))
        elif index == 7:
            available.append((value.type,))
        else:
            available.append((_broadcast(value.type),))

    candidates = QKVRoPEWithCacheCandidateProvider().get_candidates(
        DistributedCandidateContext(module, fused, PLACEMENT, tuple(available))
    )

    desired = next(
        candidate
        for candidate in candidates
        if candidate.input_types[0] == fm.TupleType((
            _head_contiguous_split(inputs[0].type.fields[0], (1,)),
            _head_contiguous_split(inputs[0].type.fields[1], (0,)),
            _head_contiguous_split(inputs[0].type.fields[2], (0,)),
        ))
    )
    broadcast = candidates[0]
    assert desired.operation_cost < broadcast.operation_cost
    assert desired.return_type.fields[0].axis_policies[1] == (
        fm.SBP.split_contiguous((1,), 1)
    )


def test_explicit_qkv_tuple_exposes_field_views_to_the_provider_search_graph():
    module, fused, inputs = _fused_module()
    qkv_input = inputs[0]
    q, k, v = tuple(
        _var(role, field)
        for role, field in zip(("q", "k", "v"), qkv_input.type.fields)
    )
    qkv = fm.Node(
        qkv_input.id,
        "builtin.tuple",
        (q.id, k.id, v.id),
        qkv_input.type,
    )
    function = module.function_map["main"]
    explicit = replace(
        module,
        nodes=(q, k, v, qkv, *inputs[1:], fused),
        functions=(replace(
            function,
            parameters=(q.id, k.id, v.id, *function.parameters[1:]),
        ),),
    )
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)

    graph = build_search_graph(
        explicit,
        PLACEMENT,
        registry,
        target.distributed_reshard_realization_policy(),
    )

    desired = fm.TupleType((
        _head_split(q.type, (1,)),
        _head_split(k.type, (0,)),
        _head_split(v.type, (0,)),
    ))
    assert any(
        candidate.return_type == desired
        for candidate in graph.bucket_map[qkv.id].candidates
    )
    assert any(
        candidate.input_types[0] == desired
        for candidate in graph.bucket_map[fused.id].candidates
    )
    sites = tuple(
        site for site in graph.reshard_sites if site.consumer_id == qkv.id
    )
    assert {site.input_index for site in sites} == {0, 1, 2}


def _fused_module():
    state_type = PagedAttentionStateConfig(
        1, 8, 128, block_size=4, num_blocks=2, lanes=8
    ).ref_type
    qkv_type = fm.TupleType(
        (
            fm.tensor_type("bfloat16", (1, 16, 128)),
            fm.tensor_type("bfloat16", (1, 8, 128)),
            fm.tensor_type("bfloat16", (1, 8, 128)),
        )
    )
    input_types = (
        qkv_type,
        fm.tensor_type("bfloat16", (128,)),
        fm.tensor_type("bfloat16", (128,)),
        fm.tensor_type("bfloat16", (128,)),
        fm.tensor_type("bfloat16", (128,)),
        fm.tensor_type("float32", (1, 1, 128)),
        fm.tensor_type("float32", (1, 1, 128)),
        state_type,
        fm.tensor_type("int32", ()),
        fm.tensor_type("bool", ()),
    )
    inputs = tuple(_var(f"arg{index}", value) for index, value in enumerate(input_types))
    attrs = {
        "q_axis": -1,
        "q_epsilon": 1e-6,
        "q_use_mean": False,
        "k_axis": -1,
        "k_epsilon": 1e-6,
        "k_use_mean": False,
        "qkv_layout": ("seq", "head", "dim"),
        "attention_layout": ("seq", "head", "dim"),
    }
    definition = fm.get_definition("nn.qkv_rope_with_cache")
    output_type = definition.infer_type(inputs, attrs)
    fused = fm.Node(
        "fused",
        "nn.qkv_rope_with_cache",
        tuple(value.id for value in inputs),
        output_type,
        definition.infer_effect(inputs, attrs),
        attrs,
    )
    module = fm.IRModule(
        "nn",
        "stats_combined",
        (*inputs, fused),
        (fm.Function("main", tuple(value.id for value in inputs), (fused.id,)),),
        "main",
    )
    return module, fused, inputs


def _head_split(value_type, axes):
    return fm.DistributedType(
        value_type,
        (
            fm.SBP.broadcast(),
            fm.SBP.split_block_cyclic(axes, 1),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )


def _head_contiguous_split(value_type, axes):
    extent = value_type.shape[1].fixed_value
    shards = 1
    for axis in axes:
        shards *= PLACEMENT.hierarchy[axis]
    return fm.DistributedType(
        value_type,
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous(axes, extent // shards),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )


def _broadcast(value_type):
    if not isinstance(value_type, fm.TensorType):
        return value_type
    return fm.DistributedType(
        value_type,
        tuple(fm.SBP.broadcast() for _ in value_type.shape),
        PLACEMENT,
    )


def _var(name, value_type):
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})
