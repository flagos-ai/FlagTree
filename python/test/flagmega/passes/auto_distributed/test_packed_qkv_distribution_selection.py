# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    CandidateBucket,
    DistributedCandidate,
    SearchGraph,
    solve_search_graph,
)
from triton.flagmega.targets import NvidiaSm90Target


def test_solver_prefers_balanced_hybrid_when_it_pays_for_partial_combine():
    """Exercise the global decision, independent of unrelated graph costs."""

    placement = fm.Placement((8, 16), "yx", "bb")
    fields = (
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
    )
    materialized = fm.TupleType(tuple(
        fm.DistributedType(
            field,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for field in fields
    ))
    partial = fm.TupleType(tuple(
        replace(field, partial=fm.SBP.partial((0,)))
        for field in materialized.fields
    ))
    logical = fm.TupleType(fields)
    projection_node = fm.Node("qkv", "builtin.none", (), logical)
    combine_node = fm.Node("combine", "builtin.identity", ("qkv",), logical)
    module = fm.IRModule(
        "ntt",
        "unused_functions_removed",
        (projection_node, combine_node),
        (),
        "main",
    )
    objective = {
        "objective_kind": "heuristic",
        "objective_model": "flagmega.packed-qkv-local-shape-balance/v1",
        "objective_evidence": ("local-k-n-imbalance",),
    }
    direct = DistributedCandidate(
        "qkv.direct",
        materialized,
        (),
        65_536 + 6_112,
        "packed-qkv-output-sbp",
        **objective,
    )
    hybrid = DistributedCandidate(
        "qkv.hybrid",
        partial,
        (),
        65_536 + 512,
        "packed-qkv-output-K-sbp-partial",
        **objective,
    )
    direct_combine = DistributedCandidate(
        "combine.direct",
        materialized,
        (materialized,),
        0,
        "packed-qkv-combine-sbp",
    )
    hybrid_combine = DistributedCandidate(
        "combine.hybrid",
        materialized,
        (partial,),
        4_608,
        "packed-qkv-combine-sbp",
    )
    target = NvidiaSm90Target()
    graph = SearchGraph(
        module,
        placement,
        (
            CandidateBucket("qkv", (direct, hybrid), True),
            CandidateBucket("combine", (direct_combine, hybrid_combine), True),
        ),
        (),
        target.distributed_reshard_realization_policy(),
        {"qkv": 1, "combine": 1},
    )

    result = solve_search_graph(graph)

    assert result.selected["qkv"] == hybrid
    assert result.selected["combine"] == hybrid_combine
    assert result.objective == hybrid.operation_cost + hybrid_combine.operation_cost
