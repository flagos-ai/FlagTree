# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import decompose_layer_norm_rule


class _Module(fm.Module):
    def __init__(self, use_mean):
        super().__init__(dialect="nn", stage="imported", entry="main")
        self.use_mean = use_mean

    def forward(self):
        value = self.input("value", fm.tensor_type("float32", [2, 16]))
        scale = self.input("scale", fm.tensor_type("float32", [16]))
        bias = self.input("bias", fm.tensor_type("float32", [16]))
        output = fm.F.nn.layer_norm(
            value,
            scale,
            bias,
            axis=-1,
            epsilon=1e-5,
            use_mean=self.use_mean,
            name="output",
        )
        self.function("main", (value, scale, bias), (output,))


def test_decompose_layer_norm_rule_uses_pattern_match_and_preserves_semantics():
    module = _Module(True).build()
    rewritten = DataflowPass(
        "DecomposeLayerNorm", (decompose_layer_norm_rule(),)).run(module)

    assert rewritten.node_map["output"].op == "nn.norm_apply"
    stats = rewritten.node_map["output.decomposed.stats"]
    assert stats.op == "nn.norm_stats"
    assert stats.attrs == {"axis": -1, "use_mean": True}
    assert not any(node.op == "nn.layer_norm" for node in rewritten.nodes)

    inputs = {
        "value": torch.randn((2, 16), dtype=torch.float32),
        "scale": torch.randn((16,), dtype=torch.float32),
        "bias": torch.randn((16,), dtype=torch.float32),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(module, inputs)[0],
    )
