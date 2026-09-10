# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.target_independent import decompose_complex_ops


class WideGlu(fm.Module):
    def __init__(self, *, reverse=False, rounded=False, different_input=False):
        super().__init__(dialect="high_level", stage="imported", entry="main")
        self.reverse, self.rounded, self.different_input = reverse, rounded, different_input

    def forward(self):
        x = self.input("x", fm.tensor_type("bfloat16", (1, 16)))
        y = self.input("y", x.type)
        gate = self.input("gate", fm.tensor_type("bfloat16", (32, 16)))
        up = self.input("up", gate.type)
        g = fm.F.math.matmul(x, gate, transpose_b=True)
        u = fm.F.math.matmul(y if self.different_input else x, up, transpose_b=True)
        g = fm.F.math.silu(fm.F.tensors.cast(g, dtype="float32"))
        if self.rounded:
            g = fm.F.tensors.cast(fm.F.tensors.cast(g, dtype="bfloat16"), dtype="float32")
        u = fm.F.tensors.cast(u, dtype="float32")
        product = fm.F.math.mul(u, g) if self.reverse else fm.F.math.mul(g, u)
        output = fm.F.tensors.cast(product, dtype="bfloat16", name="output")
        self.function("main", (x, y, gate, up), (output,))


@pytest.mark.parametrize("reverse", [False, True])
def test_normal_pipeline_fuses_wide_glu_without_changing_rounding(reverse, tmp_path):
    source = WideGlu(reverse=reverse).build()
    result = decompose_complex_ops(source)
    assert result.node_map["output"].op == "nn.dense_matmul_glu"
    assert result.node_map["output"].attrs["round_activation"] is False
    generator = torch.Generator().manual_seed(71)
    inputs = {source.node_map[name].attrs["name"]:
              torch.randn(tuple(dim.fixed_value for dim in source.node_map[name].type.shape), generator=generator).bfloat16()
              for name in source.functions[0].parameters}
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(evaluator.run(result, inputs)[0], evaluator.run(source, inputs)[0], rtol=0, atol=0)
    assert fm.load_module(fm.emit_module(result, tmp_path / "glu.py")) == result


def test_wide_glu_does_not_change_matmul_input():
    result = decompose_complex_ops(WideGlu(different_input=True).build())
    assert result.node_map["output"].op == "tensors.cast"


def test_target_independent_round_trip_elision_exposes_wide_glu():
    result = decompose_complex_ops(WideGlu(rounded=True).build())
    assert result.node_map["output"].op == "nn.dense_matmul_glu"
