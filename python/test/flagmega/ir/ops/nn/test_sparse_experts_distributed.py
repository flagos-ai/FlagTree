# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.sparse_experts.helpers import operand_types, operands

B = fm.SBP.broadcast()
MESH = fm.Placement((2, 2), "yx", "bb")


def distributed_types(*, down=False, token=B, intermediate=B, output=B, vector=True):
    types = operand_types(tokens=4, hidden=64, intermediate=32)
    dtype = fm.vector_type("bfloat16", (2, 2)) if vector else fm.DType.BFLOAT16
    if vector:
        types["q"] = fm.tensor_type(dtype, (4, 16))
        types["activations"] = fm.tensor_type(dtype, (4, 2, 8))
    scalar_intermediate = fm.scale_split_units(intermediate, 4, 1) if vector and intermediate != B else intermediate
    policies = {
        "q": (token, B),
        "activations": (token, B, intermediate),
        "router_expert_ids": (token, B),
        "router_expert_weights": (token, B),
        "gate_weight": (B, scalar_intermediate, B),
        "up_weight": (B, scalar_intermediate, B),
        "down_weight": (B, output, scalar_intermediate),
    }
    definition = SparseExpertsDown if down else SparseExpertsGateUp
    return {
        parameter.name: fm.DistributedType(types[parameter.name], policies.get(parameter.name, (B, B)), MESH)
        for parameter in definition.input_parameters
    }


def infer(definition, types, **attrs):
    return definition.prepare(operands(definition, types), attrs).result_type


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("policy", [fm.SBP.split_contiguous((1, )), fm.SBP.split_block_cyclic((1, ), 2)])
def test_gate_up_token_and_intermediate_splits_preserve_lane_units(vector, policy):
    token = fm.SBP.split_contiguous((0, ))
    types = distributed_types(token=token, intermediate=policy, vector=vector)
    result = infer(SparseExpertsGateUp, types)
    assert result.axis_policies == (token, B, policy)
    assert result.tensor.shape == fm.tensor_type("float32", (4, 2, 8 if vector else 32)).shape
    assert result.partial is None


@pytest.mark.parametrize("vector", [False, True])
def test_down_matches_intermediate_splits_and_materializes_sum_partial(vector):
    intermediate = fm.SBP.split_block_cyclic((0, ), 2)
    output = fm.SBP.split_block_cyclic((1, ), 8)
    types = distributed_types(down=True, intermediate=intermediate, output=output, vector=vector)
    result = infer(SparseExpertsDown, types)
    expected_output = fm.SBP.split_block_cyclic((1, ), 2) if vector else output
    assert result.axis_policies == (B, expected_output)
    assert result.partial == fm.SBP.partial((0, ))


@pytest.mark.parametrize("rounding", ["round_projection", "round_weighted_output"])
def test_down_rejects_split_k_that_crosses_per_route_rounding(rounding):
    types = distributed_types(down=True, intermediate=fm.SBP.split_contiguous((0, )))
    with pytest.raises(IRSchemaError, match="rounding.*split-K"):
        infer(SparseExpertsDown, types, **{rounding: True})


def test_down_per_route_rounding_accepts_token_and_output_sharding():
    token, output = fm.SBP.split_contiguous((0, )), fm.SBP.split_contiguous((1, ))
    types = distributed_types(down=True, token=token, output=output)
    result = infer(SparseExpertsDown, types, round_weighted_output=True)
    assert result.axis_policies == (token, output)
    assert result.partial is None


@pytest.mark.parametrize("definition,parameter,policies", [
    (SparseExpertsGateUp, "q", (B, fm.SBP.split_contiguous((0, )))),
    (SparseExpertsGateUp, "router_expert_ids", (B, fm.SBP.split_contiguous((0, )))),
    (SparseExpertsGateUp, "gate_weight", (fm.SBP.split_contiguous((0, )), B, B)),
    (SparseExpertsGateUp, "up_weight", (B, fm.SBP.split_contiguous((0, )), B)),
    (SparseExpertsGateUp, "gate_input_scale", (fm.SBP.split_contiguous((0, )), B)),
    (SparseExpertsDown, "router_expert_weights", (fm.SBP.split_contiguous((0, )), B)),
    (SparseExpertsDown, "down_weight", (B, B, fm.SBP.split_contiguous((0, )))),
    (SparseExpertsDown, "down_proj_scale", (fm.SBP.split_contiguous((0, )), B)),
])
def test_sparse_stages_reject_misaligned_or_dynamic_expert_splits(definition, parameter, policies):
    types = distributed_types(down=definition is SparseExpertsDown)
    types[parameter] = fm.DistributedType(types[parameter].tensor, policies, MESH)
    with pytest.raises(IRSchemaError, match="axis policies"):
        infer(definition, types)


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
def test_sparse_stages_reject_partial_input(definition):
    types = distributed_types(down=definition is SparseExpertsDown)
    name = definition.input_parameters[0].name
    types[name] = fm.DistributedType(types[name].tensor, types[name].axis_policies, MESH, fm.SBP.partial((0, )))
    with pytest.raises(IRSchemaError, match="materialized inputs"):
        infer(definition, types)


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
def test_sparse_stages_reject_mixed_placement_and_plain_operands(definition):
    types = distributed_types(down=definition is SparseExpertsDown)
    name = definition.input_parameters[0].name
    types[name] = types[name].tensor
    with pytest.raises(IRSchemaError, match="every operand"):
        infer(definition, types)


def test_gate_up_rejects_two_roles_on_one_mesh_axis():
    policy = fm.SBP.split_contiguous((0, ))
    types = distributed_types(token=policy, intermediate=policy)
    with pytest.raises(IRSchemaError, match="disjoint"):
        infer(SparseExpertsGateUp, types)


def test_gate_up_rejects_split_boundary_cutting_vector_lane_group():
    types = distributed_types(vector=False)
    policy = fm.SBP.split_block_cyclic((0, ), 2)
    for name in ("gate_weight", "up_weight"):
        types[name] = fm.DistributedType(types[name].tensor, (B, policy, B), MESH)
    with pytest.raises(IRSchemaError, match="split units"):
        infer(SparseExpertsGateUp, types, output_dtype=fm.vector_type("bfloat16", (2, 2)))
