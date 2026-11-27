# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry,
    DistributedMaterializer,
    build_search_graph,
    function_boundary_id,
    solve_search_graph,
)
from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target

from .helpers import packed_matmul_module


def test_cp_sat_selects_explicit_reshard_programs_consumed_by_materializer():
    module = packed_matmul_module()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )

    result = solve_search_graph(graph)

    assert graph.reshard_sites
    assert result.selected_reshards
    assert ("output", function_boundary_id("main"), 0) in result.selected_reshards
    materialized = DistributedMaterializer(result, policy=target.policy_version).run()
    selected_hops = sum(len(plan.step_types) for plan in result.selected_reshards.values())
    realized = [node for node in materialized.nodes if node.op.startswith("distributed.")]
    assert realized
    assert len(realized) <= selected_hops  # identical selected paths may share one materialization
    assert (
        materialized.node_map[materialized.function_map["main"].outputs[0]].op
        == "distributed.sharded_view"
    )


def test_equal_cost_search_keeps_block_cyclic_candidates_but_prefers_contiguous():
    module = packed_matmul_module(size=1024)
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )

    output_bucket = graph.bucket_map["output"]
    assert any(
        "BlockCyclicSplit" in str(value)
        for candidate in output_bucket.candidates
        for value in (candidate.return_type, *candidate.input_types)
    )

    selected = solve_search_graph(graph).selected["output"]

    assert all(
        "BlockCyclicSplit" not in str(value)
        for value in (selected.return_type, *selected.input_types)
    )


def test_structural_tuple_keeps_exact_producer_relations_without_boxing_edges():
    tensor = fm.tensor_type("bfloat16", (1, 16))
    lhs = fm.Node("lhs", "builtin.var", (), tensor, attrs={"name": "lhs"})
    rhs = fm.Node("rhs", "builtin.var", (), tensor, attrs={"name": "rhs"})
    value = fm.Node("value", "math.add", (lhs.id, rhs.id), tensor)
    pair_type = fm.TupleType((tensor, tensor))
    pair = fm.Node("pair", "builtin.tuple", (value.id, value.id), pair_type)
    module = fm.IRModule(
        "high_level",
        "packed",
        (lhs, rhs, value, pair),
        (fm.Function("main", (lhs.id, rhs.id), (pair.id,)),),
        "main",
    )
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)

    graph = build_search_graph(
        module,
        target.distributed_placements(module)[0],
        registry,
        target.distributed_reshard_realization_policy(),
    )

    assert len(graph.bucket_map["value"].candidates) > 1
    assert len(graph.bucket_map["pair"].candidates) > 1
    assert not [site for site in graph.reshard_sites if site.consumer_id == "pair"]
