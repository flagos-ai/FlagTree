# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.inference_providers import (
    TypeInferenceCandidateProvider,
)


PLACEMENT = fm.Placement((8, 16), "yx", "bb")


def _single_call_module(
    op: str,
    inputs: tuple[fm.Node, ...],
    output_type: fm.IRType,
    *,
    attrs: dict[str, object] | None = None,
) -> tuple[fm.IRModule, fm.Node]:
    output = fm.Node("output", op, tuple(value.id for value in inputs), output_type, attrs=attrs or {})
    module = fm.IRModule(
        "high_level",
        "packed",
        (*inputs, output),
        (fm.Function("main", tuple(value.id for value in inputs), (output.id,)),),
        "main",
    )
    return module, output


def _var(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def test_unpack_and_reshape_propagate_a_qkv_output_split_to_the_head_axis():
    packed = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256))
    logical = fm.tensor_type("bfloat16", (1, 2048))
    packed_node = _var("packed", packed)
    unpack_module, unpack = _single_call_module(
        "tensors.unpack", (packed_node,), logical, attrs={"axes": (1,)}
    )
    packed_split = fm.DistributedType(
        packed,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 16)),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(
        frozenset({"tensors.unpack", "tensors.reshape"})
    )

    unpack_candidates = provider.get_candidates(
        DistributedCandidateContext(
            unpack_module,
            unpack,
            PLACEMENT,
            ((packed_split,),),
        )
    )
    unpacked_split = fm.DistributedType(
        logical,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 128)),
        PLACEMENT,
    )
    assert any(candidate.return_type == unpacked_split for candidate in unpack_candidates)

    unpacked_node = _var("unpacked", logical)
    head_type = fm.tensor_type("bfloat16", (1, 16, 128))
    reshape_module, reshape = _single_call_module(
        "tensors.reshape", (unpacked_node,), head_type, attrs={"shape": (1, 16, 128)}
    )
    reshape_candidates = provider.get_candidates(
        DistributedCandidateContext(
            reshape_module,
            reshape,
            PLACEMENT,
            ((unpacked_split,),),
        )
    )

    assert any(
        candidate.return_type == fm.DistributedType(
            head_type,
            (
                fm.SBP.broadcast(),
                fm.SBP.split_contiguous((1,), 1),
                fm.SBP.broadcast(),
            ),
            PLACEMENT,
        )
        for candidate in reshape_candidates
    )


def test_bitcast_preserves_split_units_when_element_width_changes():
    source = fm.tensor_type("float32", (2, 128))
    result = fm.tensor_type("bfloat16", (2, 256))
    source_node = _var("source", source)
    module, bitcast = _single_call_module(
        "tensors.bitcast",
        (source_node,),
        result,
        attrs={"dtype": fm.DType.BFLOAT16.value},
    )
    source_split = fm.DistributedType(
        source,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1)),
        PLACEMENT,
    )
    expected = fm.DistributedType(
        result,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 2)),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({"tensors.bitcast"}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            module,
            bitcast,
            PLACEMENT,
            ((source_split,),),
        )
    )

    assert any(
        candidate.input_types == (source_split,)
        and candidate.return_type == expected
        for candidate in candidates
    )


def test_rope_provider_preserves_head_split_and_broadcasts_rotary_inputs():
    query = fm.tensor_type("bfloat16", (1, 16, 128))
    rotary = fm.tensor_type("float32", (1, 1, 128))
    inputs = (_var("query", query), _var("cos", rotary), _var("sin", rotary))
    module, rope = _single_call_module("nn.rope", inputs, query)
    query_split = fm.DistributedType(
        query,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        PLACEMENT,
    )
    rotary_broadcast = fm.DistributedType(
        rotary,
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({"nn.rope"}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            module,
            rope,
            PLACEMENT,
            ((query_split,), (rotary_broadcast,), (rotary_broadcast,)),
        )
    )

    assert any(
        candidate.return_type == query_split
        and candidate.input_types == (query_split, rotary_broadcast, rotary_broadcast)
        for candidate in candidates
    )


def test_vectorized_rope_provider_preserves_head_split_and_physical_lanes():
    query = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16, 16))
    rotary = fm.tensor_type(fm.vector_type("float32", (2, 8)), (1, 1, 8))
    inputs = (_var("query", query), _var("cos", rotary), _var("sin", rotary))
    module, rope = _single_call_module("ntt.vectorized_rope", inputs, query)
    query_split = fm.DistributedType(
        query,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        PLACEMENT,
    )
    rotary_broadcast = fm.DistributedType(
        rotary,
        (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast()),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({"ntt.vectorized_rope"}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            module,
            rope,
            PLACEMENT,
            ((query_split,), (rotary_broadcast,), (rotary_broadcast,)),
        )
    )

    assert any(
        candidate.return_type == query_split
        and candidate.input_types == (query_split, rotary_broadcast, rotary_broadcast)
        for candidate in candidates
    )


def test_stateful_attention_providers_accept_head_split_without_target_geometry():
    query = fm.tensor_type("bfloat16", (1, 16, 128))
    state = fm.RefType("paged_attention_kv_cache")
    layer = fm.tensor_type("int32", ())
    inputs = (_var("query", query), _var("state", state), _var("layer", layer))
    module, attention = _single_call_module(
        "nn.paged_attention",
        inputs,
        query,
        attrs={
            "scale": 128**-0.5,
            "layout": ("seq", "head", "dim"),
            "hidden_size": 2048,
        },
    )
    query_split = fm.DistributedType(
        query,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 1), fm.SBP.broadcast()),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(frozenset({"nn.paged_attention"}))

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            module,
            attention,
            PLACEMENT,
            ((query_split,), (state,), (layer,)),
        )
    )

    assert any(
        candidate.return_type == query_split
        and candidate.input_types == (query_split, state, layer)
        for candidate in candidates
    )


def test_stateful_cache_update_costs_the_slots_instead_of_the_reference_handle():
    slots = fm.tensor_type("bfloat16", (1, 8, 128))
    state = fm.RefType("paged_attention_kv_cache")
    layer = fm.tensor_type("int32", ())
    advance = fm.tensor_type("bool", ())
    inputs = (
        _var("slots", slots),
        _var("state", state),
        _var("layer", layer),
        _var("advance", advance),
    )
    module, update = _single_call_module(
        "nn.update_paged_attention_kv_cache",
        inputs,
        state,
        attrs={"cache_kind": "key", "layout": ("seq", "head", "dim")},
    )
    slots_split = fm.DistributedType(
        slots,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 1), fm.SBP.broadcast()),
        PLACEMENT,
    )
    provider = TypeInferenceCandidateProvider(
        frozenset({"nn.update_paged_attention_kv_cache"})
    )

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            module,
            update,
            PLACEMENT,
            ((slots_split,), (state,), (layer,), (advance,)),
        )
    )
    broadcast = next(
        candidate
        for candidate in candidates
        if isinstance(candidate.input_types[0], fm.DistributedType)
        and isinstance(candidate.input_types[0].axis_policies[1], fm.SBPBroadCast)
    )
    sharded = next(
        candidate
        for candidate in candidates
        if candidate.input_types[0] == slots_split
    )

    assert sharded.operation_cost < broadcast.operation_cost


def test_type_inference_provider_has_no_vendor_or_machine_policy():
    source = inspect.getsource(TypeInferenceCandidateProvider).lower()
    for spelling in ("sm90", "nvidia", "warp", "num_stages", "block_k"):
        assert spelling not in source
