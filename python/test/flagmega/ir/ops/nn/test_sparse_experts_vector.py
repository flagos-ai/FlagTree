# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from python.test.flagmega.sparse_experts.helpers import build_module, operand_types, values_for


@pytest.mark.parametrize("definition", [SparseExpertsGateUp, SparseExpertsDown])
@pytest.mark.parametrize("lanes", [(2, ), (2, 2)])
def test_packed_sparse_experts_inputs_and_outputs_match_scalar_evaluation(definition, lanes):
    types = operand_types()
    output_dtype = fm.vector_type("bfloat16", lanes)

    class Packed(fm.Module):

        def forward(self):
            inputs = tuple(
                self.input(parameter.name, types[parameter.name]) for parameter in definition.input_parameters)
            packed = fm.F.tensors.pack(inputs[0], lanes=lanes, axis=-1)
            output = definition.construct(packed, *inputs[1:], output_dtype=output_dtype, name="experts")
            result = fm.F.tensors.unpack(output, axis=-1, name="unpacked")
            self.function("main", inputs, (result, ))

    packed = Packed(dialect="nn", stage="imported", entry="main").build()
    scalar = build_module(definition)
    values = values_for(scalar)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(packed, values)[0], evaluator.run(scalar, values)[0], rtol=0, atol=0)
    pattern = getattr(pm.F.nn, f"is_{definition.functional_name}")(output_dtype=output_dtype)
    assert pm.try_match_root(packed.node_map["experts"], pattern, packed) is not None
    wrong = getattr(pm.F.nn, f"is_{definition.functional_name}")(output_dtype="bfloat16")
    assert pm.try_match_root(packed.node_map["experts"], wrong, packed) is None
