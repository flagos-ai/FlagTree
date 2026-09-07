# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler


def _packed_glu_module():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (1, 128)), id="value"
            )
            gate = self.weight(
                "gate", fm.tensor_type("bfloat16", (64, 128)),
                source="memory", key="gate", id="gate",
            )
            up = self.weight(
                "up", fm.tensor_type("bfloat16", (64, 128)),
                source="memory", key="up", id="up",
            )
            output = fm.F.nn.dense_matmul_glu(
                value, gate, up, activation="silu", name="output"
            )
            self.function("main", (value,), (output,))

    return Graph().build()


def test_sm90_selects_a_real_packed_consumer_when_pipeline_tiles_do_not_fit():
    selected = Compiler().compile(
        _packed_glu_module(), stop_after="propose-tir"
    ).module
    point = next(
        value for value in selected.selection_points if value.id == "tir.output"
    )

    assert selected.node_map["output"].op == "nn.packed_dense_matmul_glu"
    assert selected.selection_map[point.id].candidate_id == (
        "tir.dense_matmul_glu.packed_k_major_gemv_tn16"
    )
    assert {
        candidate.id for candidate in point.candidates
    } == {
        "tir.dense_matmul_glu.packed_k_major_gemv_tn16",
        "tir.dense_matmul_glu.packed_k_major_gemv_tn32",
        "tir.dense_matmul_glu.packed_k_major_gemv_tn64",
    }
    assert all(
        candidate.parameters["packed_layout"] == "k_major_n8_k16"
        for candidate in point.candidates
    )
