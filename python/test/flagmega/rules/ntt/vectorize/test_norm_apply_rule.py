# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter, RewriteRule
from triton.flagmega.rules.ntt.vectorize import VectorizeNormApply


class _ApplyModule(fm.Module):
    def __init__(self, extent=16):
        super().__init__(dialect="nn", stage="decomposed", entry="main")
        self.extent = extent

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [2, self.extent]))
        scale = self.input("scale", fm.tensor_type("bfloat16", [self.extent]))
        bias = self.input("bias", fm.tensor_type("bfloat16", [self.extent]))
        stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=True, name="stats")
        output = fm.F.nn.norm_apply(
            value,
            stats,
            scale,
            bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=True,
            name="output",
        )
        self.function("main", (value, scale, bias), (output,))


def test_norm_apply_rule_packs_input_scale_bias_but_not_stats():
    module = _ApplyModule().build()
    rule = VectorizeNormApply()
    candidate = rule.candidates(module.node_map["output"], module)[0]
    result = rule.rewrite(module.node_map["output"], module, candidate)

    compute = next(node for node in result.prefix_nodes if node.op == "nn.norm_apply")
    assert compute.inputs[1] == "stats"
    assert all(
        isinstance(next(node for node in result.prefix_nodes if node.id == input_id).type.dtype, fm.VectorType)
        for input_id in (compute.inputs[0], compute.inputs[2], compute.inputs[3])
    )


def test_norm_apply_rule_rewrite_is_evaluator_equivalent():
    module = _ApplyModule().build()
    rule = VectorizeNormApply()
    candidate = rule.candidates(module.node_map["output"], module)[0]
    rewritten = DataflowRewriter((RewriteRule(
        "VectorizeNormApply:output",
        lambda node, _: node.id == "output" and "vectorized_from" not in node.metadata,
        lambda node, current: rule.rewrite(node, current, candidate),
    ),), remove_unused=False).rewrite(module)

    inputs = {
        "value": torch.randn((2, 16), dtype=torch.bfloat16),
        "scale": torch.randn((16,), dtype=torch.bfloat16),
        "bias": torch.randn((16,), dtype=torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(module, inputs)[0],
    )


def test_norm_apply_rule_rejects_reduction_padding():
    module = _ApplyModule(10).build()
    assert VectorizeNormApply().candidates(module.node_map["output"], module) == ()
