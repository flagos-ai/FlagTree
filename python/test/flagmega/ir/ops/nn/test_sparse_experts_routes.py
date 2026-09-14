# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dispatch, per-route projection, and combine are independent semantic stages."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.passes.target_independent import decompose_complex_ops
from python.test.flagmega.sparse_experts.helpers import build_module, operand_types, operands, values_for


@pytest.mark.parametrize("experts,routes", [(4, 2), (1, 1), (3, 3)])
def test_decomposition_exposes_dispatch_projection_and_combine(experts, routes):
    original = build_module(types=operand_types(experts=experts, routes=routes))
    result = decompose_complex_ops(original)
    assert result.node_map["experts.dispatch"].op == "nn.sparse_experts_dispatch"
    assert result.node_map["experts.gate_up"].inputs[0] == "experts.dispatch"
    down = result.node_map["experts.down"]
    assert down.op == "nn.sparse_experts_down"
    assert down.type == fm.tensor_type("float32", (2, routes, 16))
    combine = result.node_map["experts"]
    assert combine.op == "nn.sparse_experts_combine"
    coefficients = original.node_map["experts"].inputs[2]
    assert combine.inputs == (down.id, coefficients)
    assert combine.type == original.node_map["experts"].type


def test_down_does_not_consume_coefficients_or_reduce_routes():
    inputs = operands(SparseExpertsDown)
    call = SparseExpertsDown.prepare(inputs, {})
    assert "router_expert_weights" not in {parameter.name for parameter in SparseExpertsDown.input_parameters}
    assert call.result_type == fm.tensor_type("float32", (2, 2, 16))


def test_always_active_shared_routes_keep_independent_non_normalized_gates():
    module = build_module(types=operand_types(experts=3, routes=3))
    values = values_for(module)
    values["router_expert_ids"] = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    values["router_expert_weights"] = torch.tensor([[1.0, 0.2, 0.7], [1.0, 0.9, 0.1]])
    decomposed = decompose_complex_ops(module)
    assert decomposed.node_map["experts"].op == "nn.sparse_experts_combine"
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(decomposed, values)[0], evaluator.run(module, values)[0], rtol=0, atol=0)
