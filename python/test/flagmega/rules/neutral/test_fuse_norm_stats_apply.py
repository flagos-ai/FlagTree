# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.norm_stats import fuse_norm_stats_apply
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import decompose_rms_norm_rule


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="distributed", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [2, 16]))
        weight = self.input("weight", fm.tensor_type("bfloat16", [16]))
        output = fm.F.nn.rms_norm(
            value, weight, epsilon=1e-6, weight_bias=0.0, name="output")
        self.function("main", (value, weight), (output,))


def test_fuse_norm_stats_apply_round_trips_only_proven_rms_pair():
    original = _Module().build()
    decomposed = DataflowPass(
        "Decompose", (decompose_rms_norm_rule(),)).run(original)
    fused = fuse_norm_stats_apply(decomposed)

    assert fused.node_map["output"].op == "nn.rms_norm"
    assert fused.node_map["output"].attrs == {"epsilon": 1e-6, "weight_bias": 0.0}
    assert not any(node.op in {"nn.norm_stats", "nn.norm_apply"} for node in fused.nodes)
    assert not any(node.id.startswith("output.decomposed.bias") for node in fused.nodes)

    inputs = {
        "value": torch.randn((2, 16), dtype=torch.bfloat16),
        "weight": torch.randn((16,), dtype=torch.bfloat16),
    }
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(fused, inputs)[0], evaluator.run(original, inputs)[0])


def test_fuse_norm_stats_apply_keeps_layer_norm_mean_semantics_explicit():
    class _LayerNorm(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="distributed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", [1, 8]))
            scale = self.input("scale", fm.tensor_type("float32", [8]))
            bias = self.input("bias", fm.tensor_type("float32", [8]))
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=True, name="stats")
            output = fm.F.nn.norm_apply(
                value, stats, scale, bias,
                axis=-1, epsilon=1e-5, use_mean=True, name="output")
            self.function("main", (value, scale, bias), (output,))

    module = _LayerNorm().build()
    assert fuse_norm_stats_apply(module) == module


def test_unrounded_affine_cannot_fuse_to_rounding_rms_norm():
    decomposed = DataflowPass("Decompose", (decompose_rms_norm_rule(),)).run(_Module().build())
    module = replace(decomposed, nodes=tuple(
        replace(node, attrs={**node.attrs, "round_before_scale": False})
        if node.op == "nn.norm_apply" else node for node in decomposed.nodes
    ))
    assert fuse_norm_stats_apply(module).node_map["output"].op == "nn.norm_apply"
