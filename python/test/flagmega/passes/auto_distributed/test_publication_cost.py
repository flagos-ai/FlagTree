# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry, build_search_graph, solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.realization import NttDistributedReshardRealizationPolicy
from triton.flagmega.targets import NvidiaSm90Target


def _fanout_graph(*, separate=False, reshape=False, boxing=False, invocations=1, constant=False):
    placement = fm.Placement((4,), "x", "b")
    tensor = fm.tensor_type("float32", (1, 32))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value = (builder.weight("value", broad, source="memory", key="value", id="value")
             if constant else builder.var("value", broad, id="value"))
    first = builder.call("math.silu", (value,), broad, id="producer0")
    second = builder.call("math.silu", (value,), broad, id="producer1") if separate else first
    outputs = []
    for index, source in enumerate((first, second)):
        if reshape:
            source = builder.call("tensors.reshape", (source,),
                                  fm.DistributedType(fm.tensor_type("float32", (2, 16)), broad.axis_policies, placement),
                                  id=f"view{index}", attrs={"shape": (2, 16)})
        outputs.append(builder.call("math.silu", (source,), source.type, id=f"consumer{index}"))
    builder.function("main", () if constant else (value,), tuple(outputs))
    module = fm.verify_module(builder.build(entry="main"))
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    policy = NttDistributedReshardRealizationPolicy() if boxing else target.distributed_reshard_realization_policy()
    graph = build_search_graph(module, placement, registry, policy, target.distributed_reshard_cost_model())
    graph = replace(graph, buckets=tuple(replace(bucket, candidates=tuple(
        replace(candidate, operation_cost=0) for candidate in bucket.candidates)) for bucket in graph.buckets),
        invocation_counts={node.id: invocations for node in module.nodes},
        reshard_sites=tuple(replace(site, invocation_count=invocations) for site in graph.reshard_sites))
    fixed = {}
    for bucket in graph.buckets:
        if bucket.node_id.startswith("producer"):
            selected = next(c for c in bucket.candidates if isinstance(c.return_type, fm.DistributedType)
                            and isinstance(c.return_type.axis_policies[-1], fm.SBPSplit))
        else:
            selected = next(c for c in bucket.candidates if c.return_type == module.node_map[bucket.node_id].type)
        fixed[bucket.node_id] = selected.id
    return graph, fixed


@pytest.mark.parametrize("reshape", [False, True])
@pytest.mark.parametrize("invocations", [1, 10])
def test_one_producer_publication_is_shared_by_read_only_fanout(reshape, invocations):
    graph, fixed = _fanout_graph(reshape=reshape, invocations=invocations)
    result = solve_search_graph(graph, fixed_selections=fixed)
    assert result.objective == 2200 * invocations


def test_independent_producers_do_not_share_publication():
    graph, fixed = _fanout_graph(separate=True)
    assert solve_search_graph(graph, fixed_selections=fixed).objective == 4400


def test_assembling_independent_producers_in_a_tuple_does_not_merge_origins():
    from triton.flagmega.passes.auto_distributed.publication_cost import publication_origin

    graph, _ = _fanout_graph(separate=True)
    first, second = (graph.module.node_map[f"producer{i}"] for i in range(2))
    pair = fm.Node("pair", "builtin.tuple", (first.id, second.id), fm.TupleType((first.type, second.type)))
    fields = tuple(fm.Node(f"field{i}", "builtin.get_item", (pair.id,), source.type, attrs={"index": i})
                   for i, source in enumerate((first, second)))
    function = replace(graph.module.functions[0], outputs=tuple(n.id for n in fields))
    module = fm.verify_module(replace(graph.module, nodes=(*graph.module.nodes, pair, *fields), functions=(function,)))
    graph = replace(graph, module=module)
    assert publication_origin(graph, "field0") == "producer0"
    assert publication_origin(graph, "field1") == "producer1"


def test_boxing_transfers_remain_per_edge():
    graph, fixed = _fanout_graph(boxing=True)
    # One input B->S Boxing and two output S->B Boxings, each 128 bytes.
    assert solve_search_graph(graph, fixed_selections=fixed).objective == 3 * 128 * 100


def test_tuple_value_publication_and_stats_reduce_share_source_completion():
    class StatsFanout(fm.Module):
        def forward(self):
            placement = fm.Placement((8, 16), "yx", "bb")
            broad = fm.DistributedType(fm.tensor_type("float32", (1, 128)),
                                       (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
            param = fm.DistributedType(fm.tensor_type("float32", (128,)), (fm.SBP.broadcast(),), placement)
            x = self.input("x", broad, id="x")
            residual = self.input("residual", broad, id="residual")
            scale = self.input("scale", param, id="scale")
            bias = self.input("bias", param, id="bias")
            producer = fm.F.math.silu(x, name="producer")
            combined = fm.F.ntt.add_norm_stats(producer, residual, axis=1, use_mean=False, name="combined")
            value = fm.F.tensors.get_item(combined, 0, name="value")
            stats = fm.F.tensors.get_item(combined, 1, name="stats")
            norm = fm.F.nn.norm_apply(value, stats, scale, bias, axis=1, epsilon=1e-6,
                                     use_mean=False, name="norm")
            self.function("main", (x, residual, scale, bias), (norm,))

    module = StatsFanout(dialect="ntt", stage="stats_combined", entry="main").build()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(module, fm.Placement((8, 16), "yx", "bb"), registry,
                               target.distributed_reshard_realization_policy(),
                               target.distributed_reshard_cost_model(), target.distributed_operation_cost_model())
    graph = replace(graph, buckets=tuple(replace(bucket, candidates=tuple(
        replace(c, operation_cost=0) for c in bucket.candidates)) for bucket in graph.buckets))
    split = fm.DistributedType(module.node_map["x"].type.tensor,
                               (fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 1)), graph.placement)
    producer = next(c for c in graph.bucket_map["producer"].candidates if c.return_type == split)
    combined = next(c for c in graph.bucket_map["combined"].candidates if c.input_types == (split, split))
    norm = next(c for c in graph.bucket_map["norm"].candidates if c.return_type == module.node_map["norm"].type)
    actual = solve_search_graph(graph, fixed_selections={"producer": producer.id, "combined": combined.id, "norm": norm.id})
    source = fm.Node("partial", "builtin.var", (), combined.return_type.fields[1])
    destination = norm.input_types[1]
    factors = fm.get_definition("distributed.boxing").cost_factors((source,), {"new_type": destination}, destination)
    # Both fields were written by one AddNormStats. The barrier needed before
    # reducing its stats also publishes its value; reduction traffic remains.
    assert actual.objective == graph.operation_cost_model.get_latency(factors, destination)


def test_constant_island_does_not_subtract_runtime_publication_from_offline_cost():
    from triton.flagmega.passes.auto_distributed.reshard_cost import DistributedReshardCostModel
    from triton.flagmega.passes.auto_distributed.realization import DistributedReshardSourceKind

    class OfflineConstantCost(DistributedReshardCostModel):
        def realization_cost(self, context, realization):
            if context.source_kind is DistributedReshardSourceKind.CONSTANT:
                return 0
            return super().realization_cost(context, realization)

    graph, fixed = _fanout_graph(constant=True, reshape=True)
    assert {"producer0", "view0", "view1"}.issubset(graph.constant_ids)
    result = solve_search_graph(replace(graph, reshard_cost_model=OfflineConstantCost()), fixed_selections=fixed)
    assert result.objective == 0


def test_publication_sharing_affects_solver_choice_not_only_reported_cost():
    graph, fixed = _fanout_graph()
    fixed.pop("producer0")
    buckets = tuple(replace(bucket, candidates=tuple(
        replace(c, operation_cost=3300 if c.return_type == graph.module.node_map[bucket.node_id].type else 0)
        for c in bucket.candidates)) if bucket.node_id == "producer0" else bucket for bucket in graph.buckets)
    result = solve_search_graph(replace(graph, buckets=buckets), fixed_selections=fixed)
    assert isinstance(result.selected["producer0"].return_type.axis_policies[-1], fm.SBPSplit)
    assert result.objective == 2200


def test_cost_dump_separates_shared_publication_from_per_edge_cost(tmp_path):
    from triton.flagmega.diagnostics import DumpFlags, DumpManager, DumpScope

    graph, fixed = _fanout_graph()
    with DumpScope(DumpManager(tmp_path, DumpFlags.EGRAPH_COST).root):
        solve_search_graph(graph, fixed_selections=fixed)
    picks = (tmp_path / "Costs/Pick.txt").read_text()
    assert "producer0: cost=2200 invocations=1 edges=2" in picks
    assert picks.count("edge_cost=0 invocations=1 publication=producer0") == 2
    assert "Objective : 2200" in (tmp_path / "Costs/Solve.txt").read_text()


def test_boxing_between_a_producer_and_reshape_starts_new_publication_provenance():
    from triton.flagmega.passes.auto_distributed.publication_cost import publication_origin

    graph, _ = _fanout_graph(reshape=True, boxing=True)
    assert publication_origin(graph, "view0") == "view0"
    assert publication_origin(graph, "view1") == "view1"
    alias_graph, _ = _fanout_graph(reshape=True)
    assert publication_origin(alias_graph, "view0") == "producer0"


def test_three_qkv_fields_publish_once_through_getitem_and_reshape():
    class QKVFanout(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 32)))
            weight_type = fm.vector_type("bfloat16", (8, 2, 8))
            weights = tuple(self.input(f"weight{i}", fm.tensor_type(weight_type, (2, n)))
                            for i, n in enumerate((8, 4, 4)))
            none = fm.F.builtin.none()
            qkv = fm.F.ntt.packed_qkv_parallel_linear(
                value, *weights, *((none,) * 9), num_heads=4, num_kv_heads=2,
                output_data_type="bfloat16", name="qkv")
            outputs = []
            for index in range(3):
                field = fm.F.tensors.get_item(qkv, index=index)
                view = fm.F.tensors.reshape(field, shape=(2, -1))
                outputs.append(fm.F.math.vectorized_unary(view, unary_op="silu", name=f"consumer{index}"))
            self.function("main", (value, *weights), tuple(outputs))

    module = fm.verify_module(QKVFanout().build())
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(module, fm.Placement((4,), "x", "b"), registry,
                               target.distributed_reshard_realization_policy(), target.distributed_reshard_cost_model())
    graph = replace(graph, buckets=tuple(replace(bucket, candidates=tuple(replace(c, operation_cost=0)
                    for c in bucket.candidates)) for bucket in graph.buckets))
    qkv = next(c for c in graph.bucket_map["qkv"].candidates if c.reason == "packed-qkv-output-sbp")
    fixed = {"qkv": qkv.id}
    for index in range(3):
        bucket = graph.bucket_map[f"consumer{index}"]
        fixed[bucket.node_id] = next(c.id for c in bucket.candidates if all(
            isinstance(p, fm.SBPBroadCast) for p in c.return_type.axis_policies))
    free = solve_search_graph(replace(graph, reshard_cost_model=replace(graph.reshard_cost_model, grid_synchronization_cost=0)),
                              fixed_selections=fixed)
    published = solve_search_graph(graph, fixed_selections=fixed)
    assert published.objective == free.objective + 2200
