# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateProviderRegistry, build_search_graph, solve_search_graph,
)
from triton.flagmega.passes.auto_distributed.realization import (
    DistributedReshardRealization as Realization,
    DistributedReshardRealizationContext as Context,
    DistributedReshardSourceKind as Source,
    DistributedReshardUsageKind as Usage,
    NttDistributedReshardRealizationPolicy, PyNttDistributedReshardRealizationPolicy,
)
from triton.flagmega.targets import NvidiaSm90Target


def _types():
    placement = fm.Placement((8, 16), "yx", "bb")
    tensor = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256))
    broad = fm.DistributedType(tensor, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    split = replace(broad, axis_policies=(fm.SBP.broadcast(), fm.SBP.split_contiguous((0, 1), 2)))
    return broad, split


def test_canonical_internal_value_can_publish_at_a_function_return():
    broad, split = _types()
    context = Context(split, broad, Source.INTERNAL, Usage.FUNCTION_BOUNDARY)
    assert PyNttDistributedReshardRealizationPolicy().classify(context) is Realization.SHARDED_VIEW
    assert NttDistributedReshardRealizationPolicy().classify(context) is Realization.BOXING


def test_function_return_does_not_turn_partial_or_unknown_parameter_storage_into_a_view():
    broad, split = _types()
    policy = PyNttDistributedReshardRealizationPolicy()
    assert policy.classify(Context(replace(broad, partial=fm.SBP.partial((0, 1))), broad,
                                   Source.INTERNAL, Usage.FUNCTION_BOUNDARY)) is Realization.BOXING
    assert policy.classify(Context(split, broad, Source.FUNCTION_PARAMETER,
                                   Usage.FUNCTION_BOUNDARY)) is Realization.BOXING


def residual_callee():
    class Graph(fm.Module):
        def forward(self):
            broad, _ = _types()
            x = self.input("x", broad, id="x")
            residual = self.input("residual", broad, id="residual")
            computed = fm.F.math.vectorized_unary(x, unary_op="silu", name="computed", metadata={
                "selected_vectorization": "vectorization.last_axis",
                "selected_vector_axes": (1,), "selected_vector_lanes": (8,),
            })
            combined = fm.F.ntt.add_norm_stats(computed, residual, axis=1, use_mean=False, name="combined")
            value = fm.F.tensors.get_item(combined, 0, name="value")
            stats = fm.F.tensors.get_item(combined, 1, name="stats")
            first = self.input("first", broad, id="first")
            second = self.input("second", broad, id="second")
            call = fm.F.builtin.call(first, second, result_type=combined.type, callee="layer", name="call")
            self.function("layer", (x, residual), (value, stats), attrs={"reusable": True, "noinline": True})
            self.function("main", (first, second), (call,))

    return Graph(dialect="ntt", stage="stats_combined", entry="main").build()


def test_default_search_does_not_force_add_stats_broadcast_at_a_callee_return():
    module = residual_callee()
    target = NvidiaSm90Target()
    registry = DistributedCandidateProviderRegistry()
    target.register_auto_distributed_candidate_providers(registry)
    graph = build_search_graph(module, _types()[0].placement, registry,
                               target.distributed_reshard_realization_policy(),
                               target.distributed_reshard_cost_model(), target.distributed_operation_cost_model())
    selected = solve_search_graph(graph)
    assert isinstance(selected.selected["combined"].return_type.fields[0].axis_policies[-1], fm.SBPSplit)
    assert selected.selected["combined"].return_type.fields[1].partial == fm.SBP.partial((0, 1))
