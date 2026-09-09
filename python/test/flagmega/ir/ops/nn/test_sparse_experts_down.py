# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.sparse_experts.helpers import build_module, operands, operand_types, values_for


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("round_projection,round_weighted_output", [(False, False), (True, False), (False, True),
                                                                    (True, True)])
def test_down_rounds_at_declared_boundaries_and_reduces_in_route_order(dtype, round_projection, round_weighted_output):
    attrs = dict(round_projection=round_projection, round_weighted_output=round_weighted_output)
    module = build_module(SparseExpertsDown, types=operand_types(dtype=dtype), attrs=attrs)
    values = values_for(module)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, values)[0]
    ids = values["router_expert_ids"].long()
    activation = values["activations"]
    down = values["down_weight"][ids].float()
    scale = values["down_input_scale"][ids]
    projection = torch.matmul(down, (activation.float() / scale).unsqueeze(-1)).squeeze(-1)
    projection = projection * scale * values["down_proj_scale"][ids]
    if round_projection:
        projection = projection.to(activation.dtype).float()
    weighted = projection * values["router_expert_weights"].unsqueeze(-1)
    if round_weighted_output:
        weighted = weighted.to(activation.dtype).float()
    expected = (weighted[:, 0] + weighted[:, 1]).to(activation.dtype)
    torch.testing.assert_close(actual, expected, rtol=1e-5 if dtype == "float32" else 0,
                               atol=1e-5 if dtype == "float32" else 0)


@pytest.mark.parametrize("parameter,dtype,shape", [
    ("activations", "bfloat16", (2, 3, 12)),
    ("activations", "bfloat16", (2, 2, 11)),
    ("router_expert_weights", "float32", (2, 1)),
    ("router_expert_weights", "int32", (2, 2)),
    ("down_weight", "float32", (4, 16, 12)),
    ("down_input_scale", "float32", (4, 2)),
])
def test_down_rejects_invalid_named_operand(parameter, dtype, shape):
    types = operand_types()
    types[parameter] = fm.tensor_type(dtype, shape)
    with pytest.raises(IRSchemaError):
        SparseExpertsDown.prepare(operands(SparseExpertsDown, types), {})


def test_down_weighted_rounding_is_not_algebraically_elided():
    modules = [build_module(SparseExpertsDown, attrs={"round_weighted_output": rounding}) for rounding in (False, True)]
    values = values_for(modules[0])
    evaluator = TorchEvaluator(DictWeightResolver({}))
    wide, rounded = [evaluator.run(module, values)[0] for module in modules]
    assert not torch.equal(wide, rounded)


@pytest.mark.parametrize("expert_id", [-1, 4])
def test_down_rejects_out_of_range_experts(expert_id):
    module = build_module(SparseExpertsDown)
    values = values_for(module)
    values["router_expert_ids"][0, 0] = expert_id
    with pytest.raises(EvaluationError, match="expert ids"):
        TorchEvaluator(DictWeightResolver({})).run(module, values)


def test_down_cost_is_independent_of_unselected_experts():
    costs = []
    for experts in (4, 256):
        inputs = operands(SparseExpertsDown, operand_types(experts=experts))
        call = SparseExpertsDown.prepare(inputs, {})
        costs.append(SparseExpertsDown.cost_factors(inputs, call.attrs, call.result_type))
    assert costs[0] == costs[1]
    assert costs[0].simt_fma_operations == 2 * 2 * 12 * 16
