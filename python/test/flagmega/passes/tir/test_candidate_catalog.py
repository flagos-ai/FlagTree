# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


def test_generic_tir_catalog_exposes_norm_and_matrix_variants():
    imported = fm.IRModule(
        dialect="high_level",
        stage="imported",
        nodes=(
        fm.Node(
            id="hidden",
            op="builtin.var",
            inputs=(),
            type=fm.tensor_type("bfloat16", [1, 2048]),
        ), fm.Node(
            id="gate_weight",
            op="builtin.var",
            inputs=(),
            type=fm.tensor_type("bfloat16", [6144, 2048]),
        ), fm.Node(
            id="up_weight",
            op="builtin.var",
            inputs=(),
            type=fm.tensor_type("bfloat16", [6144, 2048]),
        ), fm.Node(
            id="intermediate",
            op="builtin.var",
            inputs=(),
            type=fm.tensor_type("bfloat16", [1, 6144]),
        ), fm.Node(
            id="down_weight",
            op="builtin.var",
            inputs=(),
            type=fm.tensor_type("bfloat16", [2048, 6144]),
        ), fm.Node(
            id="input_norm",
            op="nn.rms_norm",
            inputs=(),
            type=fm.tensor_type("bfloat16", [1, 2048]),
            metadata={
                "selected_vectorization": "vectorization.norm.reduction_axis",
                "selected_vector_axes": (1,),
                "selected_vector_lanes": (8,),
            },
        ), fm.Node(
            id="mlp_gate_up",
            op="nn.dense_matmul_glu",
            inputs=("hidden", "gate_weight", "up_weight"),
            type=fm.tensor_type("bfloat16", [1, 6144]),
        ), fm.Node(
            id="mlp_down",
            op="math.matmul",
            inputs=("intermediate", "down_weight"),
            type=fm.tensor_type("bfloat16", [1, 2048]),
            metadata={
                "selected_vectorization": "vectorization.matmul.n",
                "selected_vector_axes": (1,),
                "selected_vector_lanes": (8,),
            },
        )),
        functions=(),
        entry="main",
    )

    proposed = NvidiaSm90Target().propose_tir(imported)
    glu_point = next(value for value in proposed.selection_points if value.id == "tir.mlp_gate_up")
    assert glu_point.default_candidate == "tir.dense_matmul_glu.gemv_tn16"
    assert {
        candidate.id: candidate.parameters["tile_n"]
        for candidate in glu_point.candidates
    } == {
        "tir.dense_matmul_glu.gemv_tn16": 16,
        "tir.dense_matmul_glu.gemv_tn32": 32,
        "tir.dense_matmul_glu.gemv_tn64": 64,
    }
    norm_point = next(value for value in proposed.selection_points if value.id == "tir.input_norm")
    assert norm_point.default_candidate == "tir.rms_norm.local"
    assert {candidate.id for candidate in norm_point.candidates} == {
        "tir.rms_norm.local",
    }
    assert next(
        candidate for candidate in glu_point.candidates
        if candidate.id == "tir.dense_matmul_glu.gemv_tn32"
    ).parameters["block_k"] == 256
    dense_point = next(value for value in proposed.selection_points if value.id == "tir.mlp_down")
    assert dense_point.default_candidate == "tir.dense_matmul.gemv"
    assert {candidate.id for candidate in dense_point.candidates} == {
        "tir.dense_matmul.gemv",
        "tir.dense_matmul.tensor_descriptor_gemv",
    }

    lowered = NvidiaSm90Target().lower_to_tir(proposed)
    # No selected implementation requests warp specialization, so the target
    # launch contract uses its ordinary four-warp entry.
    assert lowered.metadata["launch_contract"]["num_warps"] == 4
