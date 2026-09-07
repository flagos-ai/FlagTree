# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def _build_rms_norm(width: int):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (3, width)), id="value")
            weight = self.input("weight", fm.tensor_type("bfloat16", (width,)), id="weight")
            output = fm.F.nn.rms_norm(value, weight, epsilon=1e-6, name="output")
            self.function("main", (value, weight), (output,))

    return Graph().build()


def test_explicit_norm_does_not_pad_a_non_divisible_reduction_axis():
    module = _build_rms_norm(10)
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    assert "vectorization.output.decomposed.stats" not in vectorized.selection_map
    assert "vectorization.output" not in vectorized.selection_map
    assert vectorized.node_map["output.decomposed.stats"].op == "nn.norm_stats"
    assert vectorized.node_map["output"].op == "nn.norm_apply"

    value = torch.randn(3, 10, dtype=torch.bfloat16)
    weight = torch.randn(10, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"value": value, "weight": weight})[0],
        evaluator.run(module, {"value": value, "weight": weight})[0],
    )


def test_explicit_norm_vectorizes_stats_and_apply_on_a_divisible_reduction_axis():
    module = _build_rms_norm(16)
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module

    assert vectorized.selection_map[
        "vectorization.output.decomposed.stats"
    ].candidate_id == "vectorization.norm_stats.reduction_axis"
    assert vectorized.selection_map[
        "vectorization.output"
    ].candidate_id == "vectorization.norm_apply.reduction_axis"
    assert vectorized.node_map["output.decomposed.stats"].metadata[
        "vectorized_from"
    ] == "nn.norm_stats"
    assert vectorized.node_map["output"].metadata["vectorized_from"] == "nn.norm_apply"

    value = torch.randn(3, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"value": value, "weight": weight})[0],
        evaluator.run(module, {"value": value, "weight": weight})[0],
    )
