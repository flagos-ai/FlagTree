# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.tir import find_norm_consumer_matches


def _norm_glu_boundary(*, extra_user: bool = False):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 128)), id="value")
            scale = self.input("scale", fm.tensor_type("bfloat16", (128,)), id="scale")
            gate = self.input("gate", fm.tensor_type("bfloat16", (64, 128)), id="gate")
            up = self.input("up", fm.tensor_type("bfloat16", (64, 128)), id="up")
            norm = fm.F.nn.rms_norm(
                value, scale, epsilon=1e-6, weight_bias=0.0, name="norm")
            output = fm.F.nn.dense_matmul_glu(norm, gate, up, name="glu")
            outputs = [output]
            if extra_user:
                outputs.append(fm.F.math.add(norm, norm, name="extra"))
            self.function("main", (value, scale, gate, up), tuple(outputs))

    return Graph().build()


def test_norm_consumer_match_captures_weight_and_rounding_contract():
    match = find_norm_consumer_matches(_norm_glu_boundary())["glu"]

    assert match.producer == "norm"
    assert match.consumer == "glu"
    assert match.weight == "scale"
    assert match.epsilon == 1e-6
    assert match.weight_bias == 0.0


def test_norm_consumer_match_rejects_a_second_norm_reader():
    assert find_norm_consumer_matches(_norm_glu_boundary(extra_user=True)) == {}


def test_norm_consumer_match_accepts_explicit_external_rms_statistics():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 128)), id="value")
            stats = self.input("stats", fm.tensor_type("float32", (1, 1, 1)), id="stats")
            scale = self.input("scale", fm.tensor_type("bfloat16", (128,)), id="scale")
            gate = self.input("gate", fm.tensor_type("bfloat16", (64, 128)))
            up = self.input("up", fm.tensor_type("bfloat16", (64, 128)))
            bias = fm.F.builtin.splat_const(scale.type, 0.0, name="bias")
            norm = fm.F.nn.norm_apply(
                value, stats, scale, bias,
                axis=-1, epsilon=1e-6, use_mean=False, name="norm")
            output = fm.F.nn.dense_matmul_glu(norm, gate, up, name="glu")
            self.function("main", (value, stats, scale, gate, up), (output,))

    match = find_norm_consumer_matches(Graph().build())["glu"]
    assert match.producer == "norm"
    assert match.stats == "stats"
    assert match.weight == "scale"
