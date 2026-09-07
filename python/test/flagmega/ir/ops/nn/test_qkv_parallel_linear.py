# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


class QKVModule(fm.Module):
    def __init__(self, *, partial_scales: bool = False):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.partial_scales = partial_scales

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", (2, 16)))
        q_weight = self.input("q_weight", fm.tensor_type("bfloat16", (16, 32)))
        k_weight = self.input("k_weight", fm.tensor_type("bfloat16", (16, 16)))
        v_weight = self.input("v_weight", fm.tensor_type("bfloat16", (16, 16)))
        none = fm.F.builtin.none(name="none")
        q_input_scale = (
            self.input("q_input_scale", fm.tensor_type("float32", ()))
            if self.partial_scales
            else none
        )
        parameters = [value, q_weight, k_weight, v_weight]
        if self.partial_scales:
            parameters.append(q_input_scale)
        result = fm.F.nn.qkv_parallel_linear(
            value,
            q_weight,
            k_weight,
            v_weight,
            none,
            none,
            none,
            q_input_scale,
            none,
            none,
            none,
            none,
            none,
            num_heads=4,
            num_kv_heads=2,
            output_data_type="bfloat16",
            name="qkv",
        )
        self.function("main", tuple(parameters), (result,))


def test_qkv_parallel_linear_parameter_info_type_and_evaluator_contract():
    module = QKVModule().build()
    qkv = module.node_map["qkv"]

    assert qkv.type == fm.TupleType((
        fm.tensor_type("bfloat16", (2, 32)),
        fm.tensor_type("bfloat16", (2, 16)),
        fm.tensor_type("bfloat16", (2, 16)),
    ))
    assert [parameter.name for parameter in fm.get_definition(qkv.op).input_parameters] == [
        "input", "q_weight", "k_weight", "v_weight",
        "q_bias", "k_bias", "v_bias",
        "q_input_scale", "k_input_scale", "v_input_scale",
        "q_weight_scale", "k_weight_scale", "v_weight_scale",
    ]

    generator = torch.Generator().manual_seed(20260902)
    inputs = {
        "value": torch.randn((2, 16), generator=generator).to(torch.bfloat16),
        "q_weight": torch.randn((16, 32), generator=generator).to(torch.bfloat16),
        "k_weight": torch.randn((16, 16), generator=generator).to(torch.bfloat16),
        "v_weight": torch.randn((16, 16), generator=generator).to(torch.bfloat16),
    }
    actual = TorchEvaluator(DictWeightResolver({})).run(module, inputs)[0]
    for value, name in zip(actual, ("q_weight", "k_weight", "v_weight")):
        expected = inputs["value"] @ inputs[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_qkv_parallel_linear_rejects_incomplete_scale_contract():
    with pytest.raises(IRSchemaError, match="all six scales"):
        QKVModule(partial_scales=True).build()
