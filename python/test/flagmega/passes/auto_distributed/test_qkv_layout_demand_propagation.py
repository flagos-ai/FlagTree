# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate, DistributedCandidateProviderBase, DistributedCandidateProviderRegistry,
)
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
from triton.flagmega.passes.auto_distributed.providers import PackedQKVParallelLinearCombineCandidateProvider
from triton.flagmega.passes.auto_distributed.realization import NttDistributedReshardRealizationPolicy
from triton.flagmega.passes.auto_distributed.search import build_search_graph, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import DistributedMaterializer
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine


def test_structural_projection_does_not_query_whole_tuple_reshards(monkeypatch):
    tensor = fm.tensor_type("float32", (8,))

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", fm.TupleType((tensor, tensor)))
            self.function("main", (source,), (fm.F.tensors.get_item(source, 0),))

    queries = []
    namespace = build_search_graph.__globals__
    original = namespace["_reshard_plans"]

    def plans(source, target, *args):
        if isinstance(source, fm.TupleType):
            queries.append((source, target))
        return original(source, target, *args)

    monkeypatch.setitem(namespace, "_reshard_plans", plans)
    build_search_graph(Graph(dialect="high_level", stage="packed", entry="main").build(),
                       fm.Placement((2, 4), "yx", "bb"), DistributedCandidateProviderRegistry(),
                       NttDistributedReshardRealizationPolicy())
    assert not queries


@pytest.mark.parametrize("fix_collective", [False, True])
def test_tuple_field_consumers_request_the_collective_destination_before_reshape(fix_collective):
    mesh = fm.Placement((8, 16), "yx", "bb")
    dtype = fm.vector_type("bfloat16", (8,))
    tensors = tuple(fm.tensor_type(dtype, (1, heads * 32)) for heads in (16, 2, 2))
    partial = fm.TupleType(tuple(fm.DistributedType(tensor, (
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), block),
    ), mesh, partial=fm.SBP.partial((0,))) for tensor, block in zip(tensors, (8, 4, 4))))
    wanted = tuple(fm.DistributedType(fm.tensor_type(dtype, (1, heads, 32)), (
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((axis,), 1), fm.SBP.broadcast(),
    ), mesh) for heads, axis in zip((16, 2, 2), (1, 0, 0)))
    flat = fm.TupleType(tuple(fm.DistributedType(tensor, (
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((axis,), 32),
    ), mesh) for tensor, axis in zip(tensors, (1, 0, 0))))

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", fm.TupleType(tensors))
            combined = fm.F.ntt.packed_qkv_parallel_linear_combine(source, source.type, name="combine")
            outputs = []
            for index, target in enumerate(wanted):
                value = fm.F.tensors.get_item(combined, index, name=f"field{index}")
                view = fm.F.tensors.reshape(value, shape=tuple(d.fixed_value for d in target.tensor.shape),
                                            name=f"reshape{index}")
                outputs.append(fm.F.math.vectorized_unary(view, unary_op="sigmoid", name=f"consume{index}"))
            self.function("main", (source,), tuple(outputs))

    class Source(DistributedCandidateProviderBase):
        op_names = frozenset({"builtin.var"})
        allows_partial_inputs = True
        is_exhaustive = True

        def _enumerate_candidates(self, context):
            return (DistributedCandidate("source", partial, (), 0, "test-fixed-producer"),)

    class Consumer(DistributedCandidateProviderBase):
        op_names = frozenset({"math.vectorized_unary"})
        allows_partial_inputs = False
        is_exhaustive = True

        def _enumerate_candidates(self, context):
            target = wanted[int(context.source_call.id.removeprefix("consume"))]
            return (DistributedCandidate("consumer", target, (target,), 0, "test-fixed-consumer"),)

    module = Graph(dialect="high_level", stage="packed", entry="main").build()
    registry = DistributedCandidateProviderRegistry()
    registry.add(Source())
    registry.add(Consumer())
    registry.add(PackedQKVParallelLinearCombineCandidateProvider())
    registry.add(TypeInferenceCandidateProvider(frozenset({"tensors.reshape"})))
    machine = NvidiaSm90Machine()
    graph = build_search_graph(module, mesh, registry, NttDistributedReshardRealizationPolicy(),
                               reshard_cost_model=machine.distributed_reshard_cost_model(),
                               operation_cost_model=machine.distributed_operation_cost_model())
    candidates = [c for c in graph.bucket_map["combine"].candidates if c.return_type == flat]
    assert candidates, "Consumer head ownership must reach the tuple-producing collective"
    assert not any(module.node_map[site.consumer_id].op == "builtin.get_item"
                   for site in graph.reshard_sites if site.consumer_id in module.node_map)
    selected = solve_search_graph(graph, fixed_selections={"combine": candidates[0].id} if fix_collective else None)
    result = fm.verify_module(DistributedMaterializer(selected, policy="test").run())
    for index, target in enumerate(wanted):
        view = result.node_map[f"reshape{index}"]
        assert view.type == target
        assert view.inputs == (f"field{index}",)
        assert result.node_map[f"consume{index}"].inputs == (view.id,)
        assert result.node_map[f"field{index}"].inputs == ("combine",)


def test_unavailable_tuple_destination_reshards_only_the_projected_field():
    mesh = fm.Placement((2, 4), "yx", "bb")
    tensor = fm.tensor_type("float32", (8,))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(),), mesh)
    target = fm.DistributedType(tensor, (fm.SBP.split_contiguous((1,), 2),), mesh)

    class Graph(fm.Module):
        def forward(self):
            source = self.input("source", fm.TupleType((tensor, tensor)))
            field = fm.F.tensors.get_item(source, 1, name="field")
            consumed = fm.F.math.silu(field, name="consume")
            self.function("main", (source,), (consumed,))

    class Source(DistributedCandidateProviderBase):
        op_names = frozenset({"builtin.var"})
        allows_partial_inputs = False
        is_exhaustive = True

        def _enumerate_candidates(self, context):
            return (DistributedCandidate("source", fm.TupleType((broad, broad)), (), 0, "fixed-tuple-abi"),)

    class Consumer(DistributedCandidateProviderBase):
        op_names = frozenset({"math.silu"})
        allows_partial_inputs = False
        is_exhaustive = True

        def _enumerate_candidates(self, context):
            return (DistributedCandidate("consume", target, (target,), 0, "fixed-consumer-layout"),)

    module = Graph(dialect="high_level", stage="packed", entry="main").build()
    registry = DistributedCandidateProviderRegistry()
    registry.add(Source())
    registry.add(Consumer())
    graph = build_search_graph(module, mesh, registry, NttDistributedReshardRealizationPolicy())
    assert not any(site.consumer_id == "field" for site in graph.reshard_sites)
    assert any(site.producer_id == "field" and site.consumer_id == "consume" for site in graph.reshard_sites)
    result = fm.verify_module(DistributedMaterializer(solve_search_graph(graph), policy="test").run())
    assert result.node_map["field"].type == broad
    adapter = result.node_map[result.node_map["consume"].inputs[0]]
    assert adapter.inputs == ("field",)
    assert adapter.type == target
