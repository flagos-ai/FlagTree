# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Distribution preserves each material partition until explicit reduction."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.partitioned_sparse_experts_down import PartitionedSparseExpertsDown
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider

from python.test.flagmega.ir.ops.nn.test_partitioned_sparse_experts_down import PartitionedProjection
from python.test.flagmega.passes.auto_distributed.test_sparse_experts_providers import candidates_for


def context_for(lanes, experts=3):
    module = PartitionedProjection(lanes=lanes, tokens=2, hidden=64, experts=experts).build()
    node = module.node_map["projection"]
    return DistributedCandidateContext(
        module, node, fm.Placement((4, 8), "cy", "cb"),
        tuple((module.node_map[name].type,) for name in node.inputs),
        PyNttDistributedSplitCandidateProvider(128),
    )


@pytest.mark.parametrize("lanes", ((), (2, 2)))
def test_candidates_shard_partition_and_hidden_without_implicit_reduction(lanes):
    context = context_for(lanes)
    definition = PartitionedSparseExpertsDown
    candidates = candidates_for(context)
    assert candidates
    assert any(isinstance(candidate.return_type.axis_policies[0], fm.SBPSplit)
               and candidate.return_type.axis_policies[0].hierarchy_axes == (0,)
               and isinstance(candidate.return_type.axis_policies[3], fm.SBPSplit)
               and candidate.return_type.axis_policies[3].hierarchy_axes == (1,)
               for candidate in candidates)
    b = fm.SBP.broadcast()
    for candidate in candidates:
        inputs = tuple(fm.Node(parameter.name, "builtin.var", (), kind)
                       for parameter, kind in zip(definition.input_parameters, candidate.input_types, strict=True))
        assert definition.infer_type(inputs, context.source_call.attrs) == candidate.return_type
        assert candidate.return_type.partial is None
        assert all(kind.partial is None for kind in candidate.input_types)
        assert candidate.input_types[0].axis_policies[3] == b
        assert candidate.input_types[3].axis_policies[3] == b
        assert candidate.input_types[0].axis_policies[2] == candidate.return_type.axis_policies[0]
        assert candidate.input_types[3].axis_policies[1] == candidate.return_type.axis_policies[0]
        for index in (2, 4):
            assert all(policy == b for policy in candidate.input_types[index].axis_policies)


def test_available_hidden_cyclic_policy_uses_scalar_weight_and_vector_output_units():
    context = context_for((2, 2))
    available = list(context.available_input_types)
    available[3] = (fm.DistributedType(available[3][0],
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,)), fm.SBP.split_block_cyclic((1,), 8), fm.SBP.broadcast()),
        context.placement),)
    context = replace(context, available_input_types=tuple(available))
    assert any(candidate.input_types[3] == available[3][0]
               and candidate.return_type.axis_policies[3] == fm.SBP.split_block_cyclic((1,), 2)
               for candidate in candidates_for(context))


def test_cost_counts_selected_experts_without_scanning_the_bank():
    definition = PartitionedSparseExpertsDown
    factors = []
    for experts in (3, 288):
        context = context_for((2, 2), experts=experts)
        inputs = tuple(context.module.node_map[name] for name in context.source_call.inputs)
        factors.append(definition.cost_factors(inputs, context.source_call.attrs, context.source_call.type))
    assert factors[0] == factors[1]
    assert factors[0].simt_fma_operations == 4 * 2 * 2 * 64 * 128
    assert factors[0].chip_aggregate_memory_load_bytes == 4 * 2 * 2 * 64 * 128
    assert factors[0].block_local_memory_store_bytes == 4 * 2 * 2 * 64 * 4
