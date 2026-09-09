# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.passes.auto_distributed import DistributedCandidateContext, DistributedCandidateProviderRegistry
from triton.flagmega.passes.auto_distributed.policy import NttDistributionPolicy
from triton.flagmega.targets.pyntt_split import PyNttDistributedSplitCandidateProvider
from python.test.flagmega.ir.ops.primitive_helpers import primitive_module


def candidates_for(op, types, attrs=None, available=None, mesh=(2, 2)):
    definition = fm.get_definition(op)
    module = primitive_module(definition, types, **(attrs or {}))
    node = module.node_map["output"]
    placement = fm.Placement(mesh, "xyz"[:len(mesh)], "b" * len(mesh))
    context = DistributedCandidateContext(module, node, placement, available or tuple((value, ) for value in types),
                                          PyNttDistributedSplitCandidateProvider(128))
    registry = DistributedCandidateProviderRegistry()
    NttDistributionPolicy((placement, ), context.split_candidate_provider).register_candidate_providers(registry)
    provider = registry.try_get(op)
    assert provider is not None
    candidates = provider.get_candidates(context)
    assert candidates
    for candidate in candidates:
        inputs = tuple(
            fm.Node(parameter.name, "builtin.var", (), value)
            for parameter, value in zip(definition.input_parameters, candidate.input_types))
        assert definition.infer_type(inputs, node.attrs) == candidate.return_type
    return candidates, placement


@pytest.mark.parametrize("op,attrs", [("math.sigmoid", {}), ("nn.softmax", {"axis": -1}), ("tensors.top_k", {"k": 2}),
                                      ("tensors.slice", {"starts": (0, ), "ends": (4, ), "axes": (1, )}),
                                      ("math.reduce_sum", {"axes": (-1, ), "keep_dims": True})])
@pytest.mark.parametrize("mesh", [(2, ), (2, 2), (2, 2, 2)])
def test_unary_router_providers_keep_token_sharding(op, attrs, mesh):
    candidates, _ = candidates_for(op, (fm.tensor_type("float32", (8, 8)), ), attrs, mesh=mesh)

    def tensor_output(value):
        return value.fields[0] if isinstance(value, fm.TupleType) else value

    assert any(
        isinstance(tensor_output(candidate.return_type).axis_policies[0], fm.SBPSplit) for candidate in candidates)


def test_sum_provider_preserves_explicit_partial_result():
    candidates, _ = candidates_for("math.reduce_sum", (fm.tensor_type("float32", (8, 8)), ),
                                   {"axes": (-1, ), "keep_dims": True})
    assert any(candidate.return_type.partial is not None for candidate in candidates)


def test_div_provider_aligns_both_inputs_without_forcing_replication():
    tensor = fm.tensor_type("float32", (8, 8))
    candidates, _ = candidates_for("math.div", (tensor, tensor))
    assert all(candidate.input_types[0] == candidate.input_types[1] == candidate.return_type
               for candidate in candidates)
    assert any(isinstance(candidate.return_type.axis_policies[0], fm.SBPSplit) for candidate in candidates)


def test_broadcast_provider_preserves_available_outer_split():
    tensor = fm.tensor_type("float32", (8, 1))
    candidates, _ = candidates_for("tensors.broadcast_to", (tensor, ), {"shape": (8, 4)})
    assert any(isinstance(candidate.return_type.axis_policies[0], fm.SBPSplit) for candidate in candidates)
    assert all(candidate.return_type.axis_policies[1] == fm.SBP.broadcast() for candidate in candidates)


def test_state_view_arguments_are_not_distributed_tensor_payloads():
    state = GatedDeltaNetStateConfig(3, 1, 2, 4, 4, 4, 16).ref_type
    layer = fm.tensor_type("int32", ())
    candidates, _ = candidates_for("nn.gdn_state_slice", (state, layer))
    assert len(candidates) == 1 and candidates[0].input_types == (state, layer)
    assert isinstance(candidates[0].return_type, fm.RefType)
