# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from dataclasses import replace
from collections import Counter
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel
from triton.flagmega.passes.auto_distributed.providers import (
    PackedQKVParallelLinearCombineCandidateProvider,
)


class QKVCombineModule(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="packed", entry="main")

    def forward(self):
        packed = fm.TupleType((
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 128)),
        ))
        qkv = self.input("qkv", packed)
        combine = fm.F.ntt.packed_qkv_parallel_linear_combine(
            qkv, packed, name="combine"
        )
        self.function("main", (qkv,), (combine,))


def test_demanded_combine_relation_is_costed_once_per_context(monkeypatch):
    from triton.flagmega.passes.auto_distributed import providers

    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    placement = fm.Placement((8, 16), "yx", "bb")
    source = fm.TupleType(tuple(fm.DistributedType(
        field, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)), placement,
        partial=fm.SBP.partial((0,)),
    ) for field in node.type.fields))
    target = fm.TupleType(tuple(replace(field, partial=None, axis_policies=(
        fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0,), 32),
    )) for field in source.fields))
    provider = PackedQKVParallelLinearCombineCandidateProvider()
    context = DistributedCandidateContext(module, node, placement, ((source,),))
    counted = Counter()
    original = providers._target_qkv_cost

    def cost(context, candidate):
        counted[candidate.return_type, candidate.input_types] += 1
        return original(context, candidate)

    monkeypatch.setattr(providers, "_target_qkv_cost", cost)
    for _ in range(3):
        assert target in provider.get_return_candidate_types(context, (target,))
        relations = provider.try_get_input_type_tuples(context, target)
        assert relations
        for relation in relations:
            assert provider.create_candidate(context, target, relation).return_type == target
    assert counted[target, (source,)] == 1
    fresh = replace(context)
    provider.try_get_input_type_tuples(fresh, target)
    assert counted[target, (source,)] == 2


def test_combine_provider_consumes_partial_and_rebuilds_target_output_type():
    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    placement = fm.Placement((8, 16), "yx", "bb")
    materialized = fm.TupleType(tuple(
        fm.DistributedType(
            field,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for field in node.type.fields
    ))
    partial = fm.TupleType(tuple(
        fm.DistributedType(
            field.tensor,
            field.axis_policies,
            placement,
            partial=fm.SBP.partial((0,)),
        )
        for field in materialized.fields
    ))
    context = DistributedCandidateContext(module, node, placement, ((partial,),))

    candidates = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(context)

    assert len(candidates) == 1
    assert candidates[0].input_types == (partial,)
    assert candidates[0].return_type == materialized
    assert candidates[0].target_op == "ntt.packed_qkv_parallel_linear_combine"
    assert candidates[0].target_attrs == {"output_type": materialized}
    assert PackedQKVParallelLinearCombineCandidateProvider.allows_partial_inputs


def test_combine_cost_uses_local_output_bytes_and_partial_fan_in():
    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    placement = fm.Placement((8, 16), "yx", "bb")
    broadcast = fm.TupleType(tuple(
        fm.DistributedType(
            field,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        for field in node.type.fields
    ))
    fully_partial = fm.TupleType(tuple(
        fm.DistributedType(
            field.tensor,
            field.axis_policies,
            placement,
            partial=fm.SBP.partial((0, 1)),
        )
        for field in broadcast.fields
    ))
    head_sharded = fm.TupleType(tuple(
        fm.DistributedType(
            field,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for field in node.type.fields
    ))
    hybrid_partial = fm.TupleType(tuple(
        fm.DistributedType(
            field.tensor,
            field.axis_policies,
            placement,
            partial=fm.SBP.partial((0,)),
        )
        for field in head_sharded.fields
    ))
    context = DistributedCandidateContext(
        module,
        node,
        placement,
        ((fully_partial, hybrid_partial),),
        operation_cost_model=DistributedOperationCostModel(grid_synchronization_cycles=0),
    )

    candidates = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(
        context
    )
    by_input = {candidate.input_types[0]: candidate for candidate in candidates}

    assert by_input[hybrid_partial].operation_cost == 4_608 * placement.size
    assert by_input[fully_partial].operation_cost == 1_056_768 * placement.size
    assert (
        by_input[hybrid_partial].operation_cost
        < by_input[fully_partial].operation_cost
    )


def test_combine_cost_accounts_for_one_grid_synchronization():
    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    placement = fm.Placement((8, 16), "yx", "bb")
    output = fm.TupleType(tuple(
        fm.DistributedType(
            field,
            (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)),
            placement,
        )
        for field in node.type.fields
    ))
    partial = fm.TupleType(tuple(
        fm.DistributedType(
            field.tensor,
            field.axis_policies,
            placement,
            partial=fm.SBP.partial((0,)),
        )
        for field in output.fields
    ))

    baseline = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            node,
            placement,
            ((partial,),),
            operation_cost_model=DistributedOperationCostModel(grid_synchronization_cycles=0),
        )
    )[0]
    synchronized = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            node,
            placement,
            ((partial,),),
            operation_cost_model=DistributedOperationCostModel(grid_synchronization_cycles=2200),
        )
    )[0]

    assert synchronized.operation_cost == baseline.operation_cost + 2200


def test_combine_cost_is_scaled_by_target_memory_bandwidth():
    from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel

    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    placement = fm.Placement((8, 16), "yx", "bb")
    partial = fm.TupleType(tuple(fm.DistributedType(
        t, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 8)), placement,
        partial=fm.SBP.partial((0,)),
    ) for t in node.type.fields))
    slow_model = DistributedOperationCostModel(elementwise_elements_per_cycle=1 << 40)
    context = DistributedCandidateContext(module, node, placement, ((partial,),), operation_cost_model=slow_model)
    provider = PackedQKVParallelLinearCombineCandidateProvider()
    slow, = provider.get_candidates(context)
    fast, = provider.get_candidates(replace(context, operation_cost_model=replace(
        slow_model, block_local_read_bytes_per_cycle=8, block_local_write_bytes_per_cycle=8,
        chip_global_read_bytes_per_cycle=8, chip_global_write_bytes_per_cycle=8,
    )))
    assert fast.operation_cost < slow.operation_cost
    assert fast.objective_model == slow_model.identity


def test_combine_can_materialize_consumer_demanded_output_layout():
    module = QKVCombineModule().build()
    node = module.node_map["combine"]
    mesh = fm.Placement((8, 16), "yx", "bb")
    partial = fm.TupleType(tuple(fm.DistributedType(
        t, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((1,), 4)), mesh,
        partial=fm.SBP.partial((0,)),
    ) for t in node.type.fields))
    wanted = fm.TupleType(tuple(fm.DistributedType(
        t, (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((axis,), 32)), mesh,
    ) for t, axis in zip(node.type.fields, (1, 0, 0))))
    context = DistributedCandidateContext(module, node, mesh, ((partial,),))
    provider = PackedQKVParallelLinearCombineCandidateProvider()
    assert wanted in provider.get_return_candidate_types(context, (wanted,))
    inputs = provider.try_get_input_type_tuples(context, wanted)
    assert len(inputs) == 1 and inputs[0].input_types == (partial,)
    candidate = provider.create_candidate(context, wanted, inputs[0])
    assert candidate.target_attrs == {"output_type": wanted}
    assert candidate.operation_cost > 0
