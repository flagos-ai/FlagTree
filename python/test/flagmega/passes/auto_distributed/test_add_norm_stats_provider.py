# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
import pytest
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext,
    DistributedOperationCostModel,
    AddNormStatsCandidateProvider,
)


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="stats_threaded", entry="main")

    def forward(self):
        value_type = fm.tensor_type("bfloat16", (15, 15))
        partial = self.input("partial", value_type, id="partial")
        residual = self.input("residual", value_type, id="residual")
        combine = fm.F.ntt.add_norm_stats(
            partial,
            residual,
            axis=-1,
            use_mean=False,
            name="combine",
        )
        self.function("main", (partial, residual), (combine,))


def test_prefill_owner_changes_require_an_explicit_materialized_input():
    """The 32-row residual cannot be read using its cyclic projection owners."""
    class Prefill(fm.Module):
        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (32, 256))
            projection = self.input("projection", value_type)
            residual = self.input("residual", value_type)
            result = fm.F.ntt.add_norm_stats(projection, residual, axis=-1, use_mean=False)
            self.function("main", (projection, residual), (result,))

    module = Prefill(dialect="ntt", stage="stats_threaded", entry="main").build()
    node = next(n for n in module.nodes if n.op == "ntt.add_norm_stats")
    tensor = module.node_map[node.inputs[0]].type
    placement = fm.Placement((8, 16), "yx", "bb")
    partial = fm.DistributedType(tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 1)), placement,
        partial=fm.SBP.partial((0,)))
    residual = fm.DistributedType(tensor,
        (fm.SBP.split_block_cyclic((0,), 4), fm.SBP.split_contiguous((1,), 16)), placement)
    context = DistributedCandidateContext(module, node, placement, ((partial,), (residual,)))
    relations = {c.input_types for c in AddNormStatsCandidateProvider().get_candidates(context)}
    assert (partial, residual) not in relations
    assert (residual, residual) in relations


@pytest.mark.parametrize('dtype', ['bfloat16', fm.VectorType(fm.DType.FLOAT32, (4, ))])
def test_materialized_function_parameter_in_first_operand_has_legal_views(dtype):

    class Graph(fm.Module):

        def forward(self):
            a = self.input('a', fm.tensor_type(dtype, (15, 15)))
            b = self.input('b', a.type)
            c = fm.F.ntt.add_norm_stats(a, b, axis=-1, use_mean=False)
            self.function('main', (a, b), (c, ))

    module = Graph(dialect='ntt', stage='stats_threaded', entry='main').build()
    node = next(n for n in module.nodes if n.op == 'ntt.add_norm_stats')
    value = module.node_map[node.inputs[0]].type
    placement = fm.Placement((3, 5), 'ab', 'bb')
    broadcast = fm.DistributedType(value, (fm.SBP.broadcast(), fm.SBP.broadcast()), placement)
    context = DistributedCandidateContext(module, node, placement, ((value, ), (broadcast, )))
    candidates = AddNormStatsCandidateProvider().get_candidates(context)
    assert any(c.input_types == (broadcast, broadcast) and c.return_type.fields[0] == broadcast for c in candidates)


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
               for c in AddNormStatsCandidateProvider().get_candidates(context))


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

    candidates = AddNormStatsCandidateProvider().get_candidates(context)

    relations = {candidate.input_types for candidate in candidates}
    assert (partial, broadcast) in relations
    assert (partial, split_rows) not in relations
    assert (split_rows, split_rows) in relations
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
        candidate.objective_model == "flagmega.target-op-cost.unit/v2"
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

    candidates = AddNormStatsCandidateProvider().get_candidates(context)

    assert all(
        candidate.input_types != (partial, incompatible)
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

    candidates = AddNormStatsCandidateProvider().get_candidates(context)

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

    candidates = AddNormStatsCandidateProvider().get_candidates(context)

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
            combine = fm.F.ntt.add_norm_stats(
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

    candidates = AddNormStatsCandidateProvider().get_candidates(context)
    by_inputs = {candidate.input_types: candidate for candidate in candidates}

    # The owner-preserving collective reads all 128 partials. A changed
    # owner layout exposes Boxing on its edge and costs only the local fusion
    # here; communication must not be claimed by an unrealizable fused call.
    assert by_inputs[(partial, broadcast)].operation_cost == 38497
    assert by_inputs[(broadcast, broadcast)].operation_cost == 1400
    assert (partial, split) not in by_inputs
    assert by_inputs[(split, split)].operation_cost < 1400
    assert all(
        candidate.objective_model == "test.h800-target-op-cost/v1"
        for candidate in by_inputs.values()
    )
