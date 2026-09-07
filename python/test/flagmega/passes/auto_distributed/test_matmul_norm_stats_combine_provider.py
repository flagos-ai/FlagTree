# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext,
    DistributedOperationCostModel,
    MatMulNormStatsCombineCandidateProvider,
)


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="stats_threaded", entry="main")

    def forward(self):
        value_type = fm.tensor_type("bfloat16", (15, 15))
        partial = self.input("partial", value_type, id="partial")
        residual = self.input("residual", value_type, id="residual")
        combine = fm.F.ntt.matmul_norm_stats_combine(
            partial,
            residual,
            axis=-1,
            use_mean=False,
            name="combine",
        )
        self.function("main", (partial, residual), (combine,))


def test_combine_can_request_broadcast_addend_in_the_projections_cyclic_layout():
    module = _Module().build()
    placement = fm.Placement((3, 5), "ab", "bb")
    value = module.node_map["partial"].type
    projection = fm.DistributedType(value,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 1)), placement)
    broadcast = fm.DistributedType(value, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    context = DistributedCandidateContext(module, module.node_map["combine"], placement, ((projection,), (broadcast,)))
    assert any(c.input_types == (projection, projection) and c.return_type.fields[0] == projection
               and c.return_type.fields[1].partial == fm.SBP.partial((0, 1))
               for c in MatMulNormStatsCombineCandidateProvider().get_candidates(context))


def test_provider_exposes_partial_to_materialized_tuple_relations_on_generic_mesh():
    module = _Module().build()
    placement = fm.Placement((3, 5), "ab", "bb")
    value_type = module.node_map["partial"].type
    partial = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    broadcast = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split_rows = fm.DistributedType(
        value_type,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
        placement,
    )
    context = DistributedCandidateContext(
        module,
        module.node_map["combine"],
        placement,
        ((partial,), (broadcast, split_rows)),
    )

    candidates = MatMulNormStatsCombineCandidateProvider().get_candidates(context)

    relations = {candidate.input_types for candidate in candidates}
    assert (partial, broadcast) in relations
    assert (partial, split_rows) in relations
    assert {broadcast, split_rows}.issubset({
        candidate.return_type.fields[0] for candidate in candidates
    })
    assert all(
        candidate.return_type.fields[1].partial is None
        for candidate in candidates
        if candidate.return_type.fields[0] in {broadcast, split_rows}
    )
    assert all(candidate.objective_kind == "analytic" for candidate in candidates)
    assert all(
        candidate.objective_model == "flagmega.target-op-cost.unit/v1"
        for candidate in candidates
    )


def test_provider_rejects_partial_input_when_addend_split_uses_nonpartial_axis():
    module = _Module().build()
    placement = fm.Placement((3, 5), "ab", "bb")
    value_type = module.node_map["partial"].type
    partial = fm.DistributedType(
        value_type,
        (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((1,)),
    )
    incompatible = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,))),
        placement,
    )
    context = DistributedCandidateContext(
        module,
        module.node_map["combine"],
        placement,
        ((partial,), (incompatible,)),
    )

    candidates = MatMulNormStatsCombineCandidateProvider().get_candidates(context)

    assert all(
        candidate.input_types[1] != incompatible
        for candidate in candidates
    )


def test_provider_materializes_a_logical_function_parameter_at_its_boundary():
    module = _Module().build()
    placement = fm.Placement((3, 5), "ab", "bb")
    value_type = module.node_map["partial"].type
    partial = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    context = DistributedCandidateContext(
        module,
        module.node_map["combine"],
        placement,
        ((partial,), (value_type,)),
    )

    candidates = MatMulNormStatsCombineCandidateProvider().get_candidates(context)

    assert any(
        isinstance(candidate.input_types[1], fm.DistributedType)
        and all(
            isinstance(policy, fm.SBPBroadCast)
            for policy in candidate.input_types[1].axis_policies
        )
        and candidate.input_types[0] == partial
        for candidate in candidates
    )


def test_provider_exposes_storage_only_split_views_for_a_broadcast_addend():
    module = _Module().build()
    placement = fm.Placement((3, 5), "ab", "bb")
    value_type = module.node_map["partial"].type
    partial = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    broadcast = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    context = DistributedCandidateContext(
        module,
        module.node_map["combine"],
        placement,
        ((partial,), (broadcast,)),
    )

    candidates = MatMulNormStatsCombineCandidateProvider().get_candidates(context)

    split = next(
        candidate
        for candidate in candidates
        if isinstance(candidate.input_types[1].axis_policies[-1], fm.SBPSplit)
        and candidate.input_types[1].axis_policies[-1].hierarchy_axes == (0, 1)
    )
    assert split.return_type.fields[0] == split.input_types[1]
    assert split.return_type.fields[1].partial == fm.SBP.partial((0, 1))


def test_provider_uses_hierarchical_target_cost_for_materialized_value_layouts():
    class _VectorModule(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="stats_threaded", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256))
            partial = self.input("partial", value_type, id="partial")
            residual = self.input("residual", value_type, id="residual")
            combine = fm.F.ntt.matmul_norm_stats_combine(
                partial,
                residual,
                axis=-1,
                use_mean=False,
                name="combine",
            )
            self.function("main", (partial, residual), (combine,))

    module = _VectorModule().build()
    placement = fm.Placement((8, 16), "yx", "bb")
    value_type = module.node_map["partial"].type
    partial = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
        partial=fm.SBP.partial((0, 1)),
    )
    broadcast = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split = fm.DistributedType(
        value_type,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 2)),
        placement,
    )
    model = DistributedOperationCostModel(
        block_local_read_bytes_per_cycle=1024,
        block_local_write_bytes_per_cycle=1024,
        block_local_latency_cycles=20,
        elementwise_elements_per_cycle=128,
        simt_fma_per_cycle=64,
        chip_global_read_bytes_per_cycle=1908,
        chip_global_write_bytes_per_cycle=1908,
        chip_global_latency_cycles=300,
        block_synchronization_cycles=25,
        grid_synchronization_cycles=2200,
        identity="test.h800-target-op-cost/v1",
    )
    context = DistributedCandidateContext(
        module,
        module.node_map["combine"],
        placement,
        ((partial,), (broadcast, split)),
        operation_cost_model=model,
    )

    candidates = MatMulNormStatsCombineCandidateProvider().get_candidates(context)
    by_value_layout = {
        candidate.return_type.fields[0]: candidate for candidate in candidates
    }

    # These values reproduce nncase's factor aggregation: vector lanes count
    # in memory traffic, the outer vector shape counts in CPU work, and the
    # partial materialization serializes one 2200-cycle grid barrier.
    assert by_value_layout[broadcast].operation_cost == 3600
    assert by_value_layout[split].operation_cost == 2782
    assert all(
        candidate.objective_model == "test.h800-target-op-cost/v1"
        for candidate in by_value_layout.values()
    )
