# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry, build_search_graph, solve_search_graph,
)
from triton.flagmega.targets import NvidiaSm90Target


class ProjectionNorm(fm.Module):
    def __init__(self, *, shared=False, width=2048, reduction=4096):
        super().__init__(dialect="ntt", stage="stats_combined", entry="main")
        self.shared, self.width, self.reduction = shared, width, reduction

    def forward(self):
        lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, self.reduction)))
        rhs = self.input("rhs", fm.tensor_type(fm.vector_type("bfloat16", (8, 2, 8)),
                                              (self.reduction // 16, self.width // 8)))
        residual = self.input("residual", fm.tensor_type(fm.vector_type("bfloat16", (8,)),
                                                        (1, self.width // 8)))
        scale = self.input("scale", fm.tensor_type(fm.vector_type("float32", (8,)), (self.width // 8,)))
        bias = self.input("bias", scale.type)
        none = fm.F.builtin.none()
        projection = fm.F.ntt.packed_matmul(lhs, rhs, none, none, output_data_type="bfloat16", name="projection")
        combined = fm.F.ntt.add_norm_stats(projection, residual, axis=1, use_mean=False, name="combined")
        value = fm.F.tensors.get_item(combined, index=0, name="value")
        stats = fm.F.tensors.get_item(combined, index=1, name="stats")
        norm = fm.F.nn.norm_apply(value, stats, scale, bias, axis=1, epsilon=1e-6,
                                 use_mean=False, name="norm")
        self.function("main", (lhs, rhs, residual, scale, bias),
                      (value, norm, projection) if self.shared else (value, norm))


def projection_search(**kwargs):
    module = ProjectionNorm(**kwargs).build()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    return build_search_graph(module, fm.Placement((8, 16), "yx", "bb"), registry,
                              target.distributed_reshard_realization_policy(),
                              target.distributed_reshard_cost_model(),
                              target.distributed_operation_cost_model())


def test_projection_epilogue_has_operation_owned_joint_cost():
    graph = projection_search()
    projection = next(c for c in graph.bucket_map["projection"].candidates
                      if isinstance(c.return_type, fm.DistributedType)
                      and c.return_type.partial is None
                      and c.return_type.axis_policies[-1] == fm.SBP.split_contiguous((0, 1), 2))
    combined = next(c for c in graph.bucket_map["combined"].candidates
                    if c.input_types == (projection.return_type, projection.return_type))
    nodes = graph.module.node_map
    inputs = tuple(replace(nodes[value], type=value_type)
                   for value, value_type in zip(nodes["projection"].inputs, projection.input_types))
    addend = replace(nodes[nodes["combined"].inputs[1]], type=combined.input_types[1])
    definition = fm.get_definition("ntt.matmul_norm_stats")
    attrs = definition.normalize_attrs({"rhs_layout": "k_major", "axis": 1, "use_mean": False,
                                        "output_data_type": "bfloat16"})
    factors = definition.cost_factors((*inputs[:2], addend), attrs, combined.return_type)
    assert factors is not None
    projection_factors = fm.get_definition("ntt.packed_matmul").cost_factors(
        inputs, nodes["projection"].attrs, projection.return_type)
    assert factors.simt_fma_operations == projection_factors.simt_fma_operations > 0
    assert factors.block_local_memory_load_bytes > projection_factors.block_local_memory_load_bytes
    assert factors.grid_synchronizations == 0
    assert graph.operation_cost_model.get_latency(factors, combined.return_type) < (
        projection.operation_cost + combined.operation_cost)


def test_default_search_prices_the_realized_projection_epilogue():
    graph = projection_search()
    result = solve_search_graph(graph)
    combined = result.selected["combined"]
    assert isinstance(combined.return_type.fields[0].axis_policies[-1], fm.SBPSplit)
    assert combined.input_types[0] == result.selected["projection"].return_type
    assert combined.return_type.fields[1].partial is not None
    assert result.fusions


def test_joint_region_replaces_both_costs_in_solver_objective(monkeypatch, tmp_path):
    from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope

    graph = projection_search()
    result = solve_search_graph(graph)
    fixed = {node_id: candidate.id for node_id, candidate in result.selected.items()}
    standalone = replace(graph)
    # Counterfactual baseline with identical layouts and edges, not a compiler
    # option: only the accounting of this typed region differs.
    monkeypatch.setitem(standalone.__dict__, "fusions", ())
    baseline = solve_search_graph(standalone, fixed_selections=fixed)
    assert baseline.objective - result.objective == sum(
        (fusion.standalone_cost - fusion.operation_cost) * fusion.invocation_count
        for fusion in result.fusions)
    with DumpScope(DumpManager(tmp_path, DumpFlags.EGRAPH_COST).root):
        solve_search_graph(graph, fixed_selections=fixed)
    report = (tmp_path / "Costs/Pick.txt").read_text()
    assert "Realized fusions (replace member operation costs):" in report
    assert "projection,combined -> ntt.matmul_norm_stats:" in report


def test_shared_projection_cannot_claim_epilogue_savings():
    assert not projection_search(shared=True).fusions


def test_publication_and_partial_reduction_are_not_local_epilogues():
    graph = projection_search()
    fused = solve_search_graph(graph)
    broadcast = next(c for c in graph.bucket_map["combined"].candidates
                     if isinstance(c.input_types[0], fm.DistributedType)
                     and c.input_types[0].partial is None
                     and all(isinstance(p, fm.SBPBroadCast) for p in c.input_types[0].axis_policies))
    fixed = {"projection": fused.selected["projection"].id, "combined": broadcast.id}
    assert not solve_search_graph(graph, fixed_selections=fixed).fusions
    for fusion in graph.fusions:
        source_id, source_index = fusion.members[0]
        source = graph.bucket_map[source_id].candidates[source_index]
        assert source.return_type.partial is None


@pytest.mark.parametrize("override", ["agent", "analytic"])
def test_edited_standalone_prices_are_not_replaced(override):
    graph = projection_search()
    graph = replace(graph, buckets=tuple(
        replace(bucket, candidates=tuple(replace(c, operation_cost=c.operation_cost + 1,
                                                 objective_kind=override) for c in bucket.candidates))
        if bucket.node_id == "projection" else bucket for bucket in graph.buckets))
    assert not graph.fusions


def test_fused_region_is_weighted_by_function_invocations():
    graph = projection_search()
    result = solve_search_graph(graph)
    repeated = replace(graph,
                       invocation_counts={node.id: 3 for node in graph.module.nodes},
                       reshard_sites=tuple(replace(site, invocation_count=3) for site in graph.reshard_sites))
    actual = solve_search_graph(repeated, fixed_selections={n: c.id for n, c in result.selected.items()})
    assert actual.objective == 3 * result.objective
    assert all(fusion.invocation_count == 3 for fusion in actual.fusions)
