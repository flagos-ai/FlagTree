# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    build_search_graph,
    solve_search_graph,
)
from triton.flagmega.passes.functions import (
    static_function_invocation_counts,
    static_node_invocation_counts,
)
from triton.flagmega.targets import NvidiaSm90Target


def _two_call_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("float32", (16,))
    layer_value = builder.var("layer_value", value_type, id="layer_value")
    layer_result = builder.call(
        "math.silu", (layer_value,), value_type, id="layer_result")
    builder.function(
        "layer", (layer_value,), (layer_result,), attrs={"reusable": True})
    main_value = builder.var("main_value", value_type, id="main_value")
    call0 = builder.call(
        "builtin.call", (main_value,), value_type, id="call0",
        attrs={"callee": "layer"})
    call1 = builder.call(
        "builtin.call", (main_value,), value_type, id="call1",
        attrs={"callee": "layer"})
    pair_type = fm.TupleType((value_type, value_type))
    pair = builder.call(
        "builtin.tuple", (call0, call1), pair_type, id="pair")
    builder.function("main", (main_value,), (pair,))
    return fm.verify_module(builder.build(entry="main"))


def test_search_weights_reusable_function_body_by_static_call_count():
    module = _two_call_module()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )

    assert graph.invocation_counts["layer_result"] == 2
    result = solve_search_graph(graph)
    local_cost = result.selected["layer_result"].operation_cost
    assert local_cost > 0
    # Function-boundary reshards may add to the total, but the body alone must
    # contribute twice.  The pre-fix objective was smaller than this bound.
    assert result.objective >= local_cost * 2


def test_nested_reusable_call_counts_multiply_and_rejoin():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("float32", (16,))
    leaf_value = builder.var("leaf_value", value_type, id="leaf_value")
    leaf_result = builder.call(
        "math.silu", (leaf_value,), value_type, id="leaf_result")
    builder.function("leaf", (leaf_value,), (leaf_result,))

    middle_value = builder.var("middle_value", value_type, id="middle_value")
    leaf0 = builder.call(
        "builtin.call", (middle_value,), value_type, id="leaf0",
        attrs={"callee": "leaf"})
    leaf1 = builder.call(
        "builtin.call", (middle_value,), value_type, id="leaf1",
        attrs={"callee": "leaf"})
    middle_result = builder.call(
        "math.add", (leaf0, leaf1), value_type, id="middle_result")
    builder.function("middle", (middle_value,), (middle_result,))

    main_value = builder.var("main_value", value_type, id="main_value")
    middle_calls = tuple(
        builder.call(
            "builtin.call", (main_value,), value_type, id=f"middle{index}",
            attrs={"callee": "middle"})
        for index in range(3)
    )
    pair = builder.call(
        "math.add", middle_calls[:2], value_type, id="main_pair")
    main_result = builder.call(
        "math.add", (pair, middle_calls[2]), value_type, id="main_result")
    builder.function("main", (main_value,), (main_result,))
    module = fm.verify_module(builder.build(entry="main"))

    assert static_function_invocation_counts(module) == {
        "main": 1,
        "middle": 3,
        "leaf": 6,
    }
    node_counts = static_node_invocation_counts(module)
    assert node_counts["leaf_result"] == 6
    assert node_counts["middle_result"] == 3
    assert node_counts["main_result"] == 1


def test_solver_rejects_invocation_weighted_objective_overflow_explicitly():
    module = _two_call_module()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )
    hot = next(
        bucket
        for bucket in graph.buckets
        if any(candidate.operation_cost > 0 for candidate in bucket.candidates)
    )
    counts = dict(graph.invocation_counts)
    counts[hot.node_id] = (1 << 63) - 1

    with pytest.raises(
        IRVerificationError,
        match="objective coefficient .* signed 64-bit CP-SAT range",
    ):
        solve_search_graph(replace(graph, invocation_counts=counts))
