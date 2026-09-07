# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.passes.auto_distributed import DistributedReshardCostModel
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
        reshard_cost_model=DistributedReshardCostModel(
            grid_synchronization_cost=0
        ),
    )

    candidates = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(
        context
    )
    by_input = {candidate.input_types[0]: candidate for candidate in candidates}

    assert by_input[hybrid_partial].operation_cost == 4_608
    assert by_input[fully_partial].operation_cost == 1_056_768
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
            reshard_cost_model=DistributedReshardCostModel(
                grid_synchronization_cost=0
            ),
        )
    )[0]
    synchronized = PackedQKVParallelLinearCombineCandidateProvider().get_candidates(
        DistributedCandidateContext(
            module,
            node,
            placement,
            ((partial,),),
            reshard_cost_model=DistributedReshardCostModel(
                grid_synchronization_cost=2200
            ),
        )
    )[0]

    assert synchronized.operation_cost == baseline.operation_cost + 2200
