# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
from python.test.flagmega.sparse_experts.helpers import build_module, operand_types


def context_for(definition, *, rounded=False, vector=False, mesh=(2, 2)):
    types = operand_types(tokens=4, hidden=64, intermediate=32)
    if vector:
        types["q"] = fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (4, 16))
        types["activations"] = fm.tensor_type(fm.vector_type("bfloat16", (2, 2)), (4, 2, 8))
    attrs = {"round_weighted_output": rounded} if definition is SparseExpertsDown else {"round_projections": rounded}
    module = build_module(definition, types=types, attrs=attrs)
    node = module.node_map["experts"]
    return DistributedCandidateContext(module, node, fm.Placement(mesh, "xyz"[:len(mesh)], "b" * len(mesh)),
                                       tuple((module.node_map[name].type, ) for name in node.inputs),
                                       PyNttDistributedSplitCandidateProvider(128))


def candidates_for(context):
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((context.placement, ),
                          context.split_candidate_provider).register_candidate_providers(registry)
    return registry.try_get(context.source_call.op).get_candidates(context)


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("rounded,vector", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mesh", [(4, ), (2, 2), (2, 2, 2)])
def test_every_sparse_expert_candidate_agrees_with_type_inference(definition, rounded, vector, mesh):
    context = context_for(definition, rounded=rounded, vector=vector, mesh=mesh)
    candidates = candidates_for(context)
    assert candidates
    assert len({candidate.id for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        inputs = tuple(
            fm.Node(parameter.name, "builtin.var", (), value)
            for parameter, value in zip(definition.input_parameters, candidate.input_types))
        assert definition.infer_type(inputs, context.source_call.attrs) == candidate.return_type
        if definition is SparseExpertsDown and rounded:
            assert candidate.return_type.partial is None
        for parameter, value in zip(definition.input_parameters, candidate.input_types):
            if parameter.name.endswith("_weight"):
                assert value.axis_policies[0] == fm.SBP.broadcast()
            if parameter.name.startswith("router_"):
                assert value.axis_policies[1] == fm.SBP.broadcast()
            if parameter.name.endswith("_scale"):
                assert all(policy == fm.SBP.broadcast() for policy in value.axis_policies)


def test_gate_up_preserves_available_weight_block_cyclic_policy_in_vector_units():
    context = context_for(SparseExpertsGateUp, vector=True)
    parameter = SparseExpertsGateUp.gate_weight
    available = list(context.available_input_types)
    weight = available[parameter.input_index][0]
    available[parameter.input_index] = (fm.DistributedType(weight,
                                                           (fm.SBP.broadcast(), fm.SBP.split_block_cyclic(
                                                               (1, ), 8), fm.SBP.broadcast()), context.placement), )
    context = replace(context, available_input_types=tuple(available))
    assert any(candidate.return_type.axis_policies[2] == fm.SBP.split_block_cyclic((1, ), 2)
               for candidate in candidates_for(context))


def test_down_retains_available_activation_intermediate_reduction_policy():
    context = context_for(SparseExpertsDown, vector=True)
    available = list(context.available_input_types)
    available[0] = (fm.DistributedType(available[0][0],
                                       (fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.split_block_cyclic(
                                           (0, ), 2)), context.placement), )
    context = replace(context, available_input_types=tuple(available))
    assert any(candidate.input_types[0] == available[0][0] and candidate.return_type.partial == fm.SBP.partial((
        0, )) and candidate.input_types[SparseExpertsDown.down_weight.input_index].axis_policies[2] ==
               fm.SBP.split_block_cyclic((0, ), 8) for candidate in candidates_for(context))


def test_down_rounding_keeps_output_parallelism_on_the_whole_two_dimensional_mesh():
    context = context_for(SparseExpertsDown, rounded=True)
    candidates = candidates_for(context)
    assert all(candidate.return_type.partial is None for candidate in candidates)
    assert any(
        isinstance(candidate.return_type.axis_policies[1], fm.SBPSplit)
        and candidate.return_type.axis_policies[1].hierarchy_axes == (0, 1) for candidate in candidates)
