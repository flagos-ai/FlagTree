# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import decompose_rms_norm_rule


class _Module(fm.Module):
    def __init__(self, weight_bias, dtype="bfloat16"):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.weight_bias = weight_bias
        self.dtype = dtype

    def forward(self):
        value = self.input("value", fm.tensor_type(self.dtype, [2, 16]))
        weight = self.input("weight", fm.tensor_type(self.dtype, [16]))
        output = fm.F.nn.rms_norm(
            value,
            weight,
            epsilon=1e-6,
            weight_bias=self.weight_bias,
            name="output",
        )
        self.function("main", (value, weight), (output,))


@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_decompose_rms_norm_uses_generic_explicit_statistics(weight_bias):
    module = _Module(weight_bias).build()
    rewritten = DataflowPass(
        "DecomposeRMSNorm", (decompose_rms_norm_rule(),)).run(module)

    output = rewritten.node_map["output"]
    assert output.op == "nn.norm_apply"
    assert rewritten.node_map["output.decomposed.stats"].op == "nn.norm_stats"
    bias = rewritten.node_map["output.decomposed.bias"]
    assert bias.op == "builtin.splat_const" and bias.attrs["value"] == 0.0
    if weight_bias:
        assert rewritten.node_map["output.decomposed.scale"].op == "math.add"
        assert rewritten.node_map["output.decomposed.scale_offset"].attrs["value"] == weight_bias
    assert not any(node.op == "nn.rms_norm" for node in rewritten.nodes)

    inputs = {
        "value": torch.randn((2, 16), dtype=torch.bfloat16),
        "weight": torch.randn((16,), dtype=torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(module, inputs)[0],
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_decomposition_preserves_normalized_value_rounding(dtype, weight_bias):
    module = _Module(weight_bias, dtype).build()
    rewritten = DataflowPass("Decompose", (decompose_rms_norm_rule(),)).run(module)
    generator = torch.Generator().manual_seed(123)
    inputs = {
        "value": torch.randn((2, 16), generator=generator).to(getattr(torch, dtype)),
        "weight": torch.randn((16,), generator=generator).to(getattr(torch, dtype)),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0], evaluator.run(module, inputs)[0],
        rtol=0, atol=0,
    )
