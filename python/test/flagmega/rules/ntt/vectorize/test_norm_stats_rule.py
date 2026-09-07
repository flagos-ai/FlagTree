# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter, RewriteRule
from triton.flagmega.rules.ntt.vectorize import VectorizeNormStats


class _StatsModule(fm.Module):
    def __init__(self, extent=16):
        super().__init__(dialect="nn", stage="decomposed", entry="main")
        self.extent = extent

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [2, self.extent]))
        stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
        self.function("main", (value,), (stats,))


def test_norm_stats_rule_matches_nncase_last_reduction_axis_only():
    module = _StatsModule().build()
    rule = VectorizeNormStats()
    candidates = rule.candidates(module.node_map["stats"], module)
    assert len(candidates) == 1
    assert candidates[0].id == "vectorization.norm_stats.reduction_axis"
    assert candidates[0].axes == (1,)
    assert candidates[0].lanes == (8,)

    non_divisible = _StatsModule(10).build()
    assert rule.candidates(non_divisible.node_map["stats"], non_divisible) == ()


def test_norm_stats_rule_rewrite_is_evaluator_equivalent():
    module = _StatsModule().build()
    rule = VectorizeNormStats()
    candidate = rule.candidates(module.node_map["stats"], module)[0]
    rewritten = DataflowRewriter((RewriteRule(
        "VectorizeNormStats:stats",
        lambda node, _: node.id == "stats" and "vectorized_from" not in node.metadata,
        lambda node, current: rule.rewrite(node, current, candidate),
    ),), remove_unused=False).rewrite(module)

    root = rewritten.node_map["stats"]
    assert isinstance(rewritten.node_map[root.inputs[0]].type.dtype, fm.VectorType)
    assert root.metadata["vectorized_from"] == "nn.norm_stats"
    value = torch.randn((2, 16), dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(module, {"value": value})[0],
    )
