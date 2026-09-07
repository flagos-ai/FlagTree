# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


class _LayerNormModule(fm.Module):
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


def test_layer_norm_fused_evaluator_matches_torch_layer_norm():
    module = _LayerNormModule(True).build()
    value = torch.randn((2, 16), dtype=torch.float32)
    scale = torch.randn((16,), dtype=torch.float32)
    bias = torch.randn((16,), dtype=torch.float32)
    actual = TorchEvaluator(DictWeightResolver({})).run(
        module, {"value": value, "scale": scale, "bias": bias})[0]
    expected = torch.nn.functional.layer_norm(value, (16,), scale, bias, 1e-5)
    torch.testing.assert_close(actual, expected)


def test_layer_norm_python_checkpoint_uses_static_f_api(tmp_path):
    module = _LayerNormModule(False).build()
    checkpoint = fm.emit_module(module, tmp_path / "layer_norm.py")
    assert "F.nn.layer_norm(" in checkpoint.read_text()
    assert fm.load_module(checkpoint) == module
