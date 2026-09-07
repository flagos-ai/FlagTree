# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.target_independent import decompose_complex_ops


class _Module(fm.Module):
    def __init__(self):
        super().__init__(dialect="nn", stage="imported", entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [1, 16]))
        weight = self.input("weight", fm.tensor_type("bfloat16", [16]))
        output = fm.F.nn.rms_norm(
            value, weight, epsilon=1e-6, weight_bias=0.0, name="output")
        self.function("main", (value, weight), (output,))


def test_decompose_complex_ops_is_real_normalization_dataflow_not_stage_rename():
    result = decompose_complex_ops(_Module().build())
    assert result.node_map["output"].op == "nn.norm_apply"
    assert result.node_map["output.decomposed.stats"].op == "nn.norm_stats"
    assert result.node_map["output.decomposed.bias"].op == "builtin.splat_const"
