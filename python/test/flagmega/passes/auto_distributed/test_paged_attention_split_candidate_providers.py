# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.rules.neutral._utility import make_node
from triton.flagmega.passes.auto_distributed.paged_attention_providers import (
    PagedAttentionCombineCandidateProvider,
    PagedAttentionPartialCandidateProvider,
)


PLACEMENT = fm.Placement((8, 16), "yx", "bb")
LAYOUT = ("seq", "head", "dim")


def _var(name: str, value_type: fm.IRType) -> fm.Node:
    return fm.Node(name, "builtin.var", (), value_type, attrs={"name": name})


def _partial_module():
    query_type = fm.tensor_type("bfloat16", (1, 16, 8))
    query = _var("query", query_type)
    state = _var("state", fm.RefType("paged_attention_kv_cache"))
    layer = _var("layer", fm.tensor_type("int32", ()))
    partial = make_node(
        "ntt.paged_attention_partial",
        "partial",
        (query, state, layer),
        {
            "scale": 8**-0.5,
            "layout": LAYOUT,
            "hidden_size": 128,
            "split_hierarchy_axis": 0,
            "split_count": 8,
        },
        {},
    )
    module = fm.IRModule(
        "ntt",
        "attention_decomposed",
        (query, state, layer, partial),
        (fm.Function("main", ("query", "state", "layer"), ("partial",)),),
        "main",
    )
    return fm.verify_module(module), partial


def test_partial_provider_delegates_layout_legality_to_op_type_inference():
    module, partial = _partial_module()
    query_type = module.node_map["query"].type
    head_split = fm.DistributedType(
        query_type,
        (
            fm.SBP.broadcast(),
            fm.SBP.split_contiguous((1,), 1),
            fm.SBP.broadcast(),
        ),
        PLACEMENT,
    )
    # nncase's ParameterKind.Attribute is a graph operand, but it does not
    # participate in the distributed/SBP relation.
    layer = fm.tensor_type("int32", ())
    context = DistributedCandidateContext(
        module,
        partial,
        PLACEMENT,
        ((head_split,), (module.node_map["state"].type,), (layer,)),
    )

    candidates = PagedAttentionPartialCandidateProvider().get_candidates(context)

    selected = next(
        candidate for candidate in candidates
        if candidate.input_types == (head_split, module.node_map["state"].type, layer)
    )
    assert isinstance(selected.return_type, fm.TupleType)
    assert tuple(field.partial.reduce_op for field in selected.return_type.fields) == (
        fm.ReduceOp.MAX,
        fm.ReduceOp.SUM,
        fm.ReduceOp.SUM,
    )


def test_combine_provider_rebuilds_output_type_and_exposes_released_axis_choices():
    module, partial = _partial_module()
    query_policies = (
        fm.SBP.broadcast(),
        fm.SBP.split_contiguous((1,), 1),
        fm.SBP.broadcast(),
    )
    query = fm.DistributedType(
        module.node_map["query"].type, query_policies, PLACEMENT
    )
    layer = fm.tensor_type("int32", ())
    partial_candidates = PagedAttentionPartialCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            partial,
            PLACEMENT,
            ((query,), (module.node_map["state"].type,), (layer,)),
        )
    )
    partial_candidate = next(
        candidate
        for candidate in partial_candidates
        if candidate.input_types == (query, module.node_map["state"].type, layer)
    )
    states = partial_candidate.return_type.fields
    max_node, sum_node, acc_node = (
        _var("max_state", states[0].tensor),
        _var("sum_state", states[1].tensor),
        _var("acc_state", states[2].tensor),
    )
    output_type = fm.tensor_type("bfloat16", (1, 16, 8))
    combine = make_node(
        "ntt.paged_attention_combine",
        "combine",
        (max_node, sum_node, acc_node),
        {
            "layout": LAYOUT,
            "hidden_size": 128,
            "output_data_type": "bfloat16",
            "output_type": output_type,
            "split_hierarchy_axis": 0,
            "split_count": 8,
        },
        {},
    )
    combine_module = fm.IRModule(
        "ntt",
        "attention_decomposed",
        (max_node, sum_node, acc_node, combine),
        (fm.Function("main", tuple(node.id for node in (max_node, sum_node, acc_node)), (combine.id,)),),
        "main",
    )
    provider = PagedAttentionCombineCandidateProvider()

    candidates = provider.get_candidates(
        DistributedCandidateContext(
            combine_module,
            combine,
            PLACEMENT,
            ((states[0],), (states[1],), (states[2],)),
        )
    )

    assert candidates
    assert all(candidate.target_attrs["output_type"] == candidate.return_type for candidate in candidates)
    assert any(
        candidate.return_type.axis_policies == query_policies
        for candidate in candidates
    )
    assert any(
        any(
            isinstance(policy, fm.SBPSplit)
            and 0 in policy.hierarchy_axes
            for policy in candidate.return_type.axis_policies
        )
        for candidate in candidates
    )
    assert PagedAttentionCombineCandidateProvider.allows_partial_inputs
    assert PagedAttentionCombineCandidateProvider.is_exhaustive


def test_split_candidate_providers_are_target_neutral():
    source = inspect.getsource(
        __import__(
            "triton.flagmega.passes.auto_distributed.paged_attention_providers",
            fromlist=("PagedAttentionCombineCandidateProvider",),
        )
    ).lower()
    for spelling in ("nvidia", "sm90", "cuda", "mma", "tma", "warp"):
        assert spelling not in source
