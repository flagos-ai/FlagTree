# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native residual RMS keeps a complete reduction tree and accurate rsqrt."""

import json
from pathlib import Path

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.runtime import load


def _module(order="tile_fma", math_mode="accurate"):
    class Norm(fm.Module):
        def forward(self):
            left = self.input("left", fm.tensor_type("bfloat16", (1, 2048)))
            right = self.input("right", left.type)
            weight = self.input("weight", fm.tensor_type("float32", (2048,)))
            zero = self.input("zero", weight.type)
            value = fm.F.math.add(fm.F.tensors.cast(left, dtype="float32"),
                                  fm.F.tensors.cast(right, dtype="float32"))
            stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, reduction_order=order,
                                       reduction_group_size=8, reduction_block_size=2048)
            result = fm.F.nn.norm_apply(value, stats, weight, zero, axis=-1, epsilon=1e-6,
                                        use_mean=False, output_dtype="bfloat16", math_mode=math_mode)
            self.function("main", (left, right, weight, zero), (result,))
    return Norm(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_residual_rms_complete_tree(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[4] / "data" /
                          "native_hidden_norm_tree.json").read_text())
    left, right = (torch.tensor([fixture[k]], device="cuda", dtype=torch.bfloat16)
                   for k in ("left", "right"))
    weight = torch.tensor(fixture["weight"], device="cuda")
    zero = torch.zeros_like(weight)
    expected = torch.tensor([fixture["output"]], device="cuda", dtype=torch.bfloat16)
    module = _module()
    eager = TorchEvaluator(DictWeightResolver({})).run(module, {
        "left": left, "right": right, "weight": weight, "zero": zero})[0]
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    compiled = Compiler().compile(module).module
    runtime = load(write_artifact(compiled, tmp_path / "artifact", target="nvidia-sm90",
                                 emit_executable=True), device="cuda:0")
    try:
        runtime.prepare(left, right, weight, zero)
        actual = runtime.run(left, right, weight, zero)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert runtime.resource_report["spill_bytes"] == 0
    finally:
        runtime.close()
