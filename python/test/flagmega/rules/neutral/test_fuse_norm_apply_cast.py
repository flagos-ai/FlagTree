# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.target_independent import decompose_complex_ops


def module(dtype, result_dtype, *, shared=False, round_before_scale=False):
    class Graph(fm.Module):
        def forward(self):
            x = self.input("x", fm.tensor_type(dtype, (2, 32)), id="x")
            scale = self.input("scale", fm.tensor_type("bfloat16", (32,)), id="scale")
            bias = self.input("bias", fm.tensor_type("bfloat16", (32,)), id="bias")
            stats = fm.F.nn.norm_stats(x, axis=-1, use_mean=False, name="stats")
            norm = fm.F.nn.norm_apply(x, stats, scale, bias, axis=-1, epsilon=1e-6, use_mean=False,
                                      round_before_scale=round_before_scale, name="norm")
            output = fm.F.tensors.cast(norm, result_dtype, name="output")
            self.function("main", (x, scale, bias), (output, norm) if shared else (output,))
    return Graph(dialect="high_level", stage="imported", entry="main").build()


@pytest.mark.parametrize("source,result", [("float32", "bfloat16"), ("bfloat16", "float32")])
@pytest.mark.parametrize("round_before_scale", [False, True])
def test_output_cast_fuses_without_changing_either_rounding_boundary(tmp_path, source, result, round_before_scale):
    original = module(source, result, round_before_scale=round_before_scale)
    fused = decompose_complex_ops(original)
    output = fused.node_map["output"]
    assert output.op == "nn.norm_apply"
    assert output.attrs["output_dtype"] == result
    assert "norm" not in fused.node_map
    torch.manual_seed(91)
    values = {"x": torch.randn(2, 32).to(getattr(torch, source)),
              "scale": torch.randn(32).bfloat16(), "bias": torch.randn(32).bfloat16()}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(fused, values), evaluator.run(original, values), rtol=0, atol=0)
    path = fm.emit_module(fused, tmp_path / "norm.py")
    assert fm.load_module(path).semantic_hash == fused.semantic_hash


def test_externally_observable_norm_output_does_not_duplicate_compute():
    original = module("float32", "bfloat16", shared=True)
    fused = decompose_complex_ops(original)
    assert fused.node_map["output"].op == "tensors.cast"
    assert fused.node_map["norm"].op == "nn.norm_apply"
