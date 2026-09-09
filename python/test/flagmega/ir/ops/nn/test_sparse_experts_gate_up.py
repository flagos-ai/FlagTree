# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import EvaluationError, IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from python.test.flagmega.sparse_experts.helpers import build_module, operands, operand_types, values_for


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("round_projections,round_activation", [(False, False), (True, False), (True, True),
                                                                (False, True)])
def test_gate_up_rounding_contract_against_independent_reference(dtype, round_projections, round_activation):
    attrs = dict(round_projections=round_projections, round_activation=round_activation)
    module = build_module(SparseExpertsGateUp, types=operand_types(dtype=dtype), attrs=attrs)
    values = values_for(module)
    actual = TorchEvaluator(DictWeightResolver({})).run(module, values)[0]
    q, ids = values["q"], values["router_expert_ids"].long()
    expected = []
    for token in range(2):
        projections = []
        for stage in ("gate", "up"):
            scale = values[f"{stage}_input_scale"][ids[token]]
            weight = values[f"{stage}_weight"][ids[token]].float()
            projected = torch.bmm(weight, (q[token].float()[None, :] / scale).unsqueeze(-1)).squeeze(-1)
            projected = projected * scale * values[f"{stage}_proj_scale"][ids[token]]
            projections.append(projected.to(q.dtype).float() if round_projections else projected)
        gate, up = projections
        gate = torch.nn.functional.silu(gate)
        if round_activation:
            gate = gate.to(q.dtype).float()
        expected.append((gate * up).to(q.dtype))
    torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5 if dtype == "float32" else 0,
                               atol=1e-5 if dtype == "float32" else 0)


@pytest.mark.parametrize("parameter,dtype,shape", [
    ("router_expert_ids", "float32", (2, 2)),
    ("router_expert_ids", "int32", (3, 2)),
    ("router_expert_ids", "int32", (2, 0)),
    ("router_expert_ids", "int32", (2, 5)),
    ("q", "bfloat16", (2, 15)),
    ("up_weight", "bfloat16", (4, 10, 16)),
    ("gate_weight", "float32", (4, 12, 16)),
    ("gate_input_scale", "bfloat16", (4, 1)),
    ("gate_proj_scale", "float32", (4, )),
    ("up_input_scale", "float32", (3, 1)),
])
def test_gate_up_rejects_invalid_named_operands(parameter, dtype, shape):
    types = operand_types()
    types[parameter] = fm.tensor_type(dtype, shape)
    with pytest.raises(IRSchemaError):
        SparseExpertsGateUp.prepare(operands(SparseExpertsGateUp, types), {})


@pytest.mark.parametrize("attrs", [{"round_activation": 1}, {"round_projections": "true"}, {"output_dtype": "int32"},
                                   {"output_dtype": "float32"}, {"output_dtype": fm.vector_type("bfloat16", 5)}])
def test_gate_up_rejects_invalid_attributes(attrs):
    with pytest.raises(IRSchemaError):
        SparseExpertsGateUp.prepare(operands(SparseExpertsGateUp), attrs)


@pytest.mark.parametrize("expert_id", [-1, 4])
def test_gate_up_rejects_out_of_range_expert_id(expert_id):
    module = build_module(SparseExpertsGateUp)
    values = values_for(module)
    values["router_expert_ids"][0, 0] = expert_id
    with pytest.raises(EvaluationError, match="expert ids"):
        TorchEvaluator(DictWeightResolver({})).run(module, values)


def test_gate_up_infers_dynamic_tokens_without_redundant_chunk_attribute():
    tokens = fm.dim("tokens", minimum=1, maximum=32)
    types = operand_types(tokens=tokens)
    prepared = SparseExpertsGateUp.prepare(operands(SparseExpertsGateUp, types), {})
    assert prepared.result_type == fm.tensor_type("bfloat16", (tokens, 2, 12))


def test_gate_up_cost_charges_selected_experts_not_whole_bank():
    costs = []
    for experts in (4, 256):
        inputs = operands(SparseExpertsGateUp, operand_types(experts=experts))
        call = SparseExpertsGateUp.prepare(inputs, {})
        costs.append(SparseExpertsGateUp.cost_factors(inputs, call.attrs, call.result_type))
    assert costs[0] == costs[1]
    assert costs[0].simt_fma_operations == 2 * 2 * 2 * 12 * 16
