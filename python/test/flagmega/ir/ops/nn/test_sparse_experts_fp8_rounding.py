# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Block-FP8 storage must not add an undeclared projection rounding boundary."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp

from python.test.flagmega.codegen.triton.kernels.sparse_experts.helpers import stage_module


def projection_case(definition, *, rounded=False, input_scale=1.):
    attrs = {"round_projections" if definition is SparseExpertsGateUp else "round_projection": rounded}
    module = stage_module(definition, dtype="bfloat16", weight_dtype="float8_e4m3fn", block_scaled=True,
                          weight_block=128, hidden=128, intermediate=128, tokens=1, **attrs)
    activation = definition.input_parameters[0].name
    values = {}
    for key in module.function_map[module.entry].parameters:
        kind = fm.logical_type(module.node_map[key].type)
        shape = tuple(dim.fixed_value for dim in kind.shape)
        if key == "router_expert_ids":
            value = torch.zeros(shape, dtype=torch.int32)
        elif key.endswith("_proj_scale"):
            value = torch.full(shape, 1.00390625)
        elif key.endswith("_input_scale"):
            value = torch.full(shape, input_scale)
        else:
            value = torch.zeros(shape)
            value[..., 0] = 129. if key == activation else 1.
            value = value.to(getattr(torch, kind.dtype.value))
        values[key] = value
    return module, values


@pytest.mark.parametrize("definition", (SparseExpertsGateUp, SparseExpertsDown))
@pytest.mark.parametrize("rounded", (False, True))
@pytest.mark.parametrize("input_scale", (1., 1.1))
def test_fp8_projection_rounds_only_when_its_stage_requests_it(definition, rounded, input_scale):
    module, values = projection_case(definition, rounded=rounded, input_scale=input_scale)
    output, = TorchEvaluator(DictWeightResolver({})).run(module, values)
    wide = torch.tensor(129.50390625) * input_scale
    projected = wide.bfloat16().float() if rounded else wide
    expected = (projected.square().bfloat16().item() if definition is SparseExpertsGateUp else projected.item())
    torch.testing.assert_close(output, torch.full_like(output, expected), atol=0, rtol=0)
