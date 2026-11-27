# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes.tir import find_projection_logits_argmax_matches
from triton.flagmega.targets import NvidiaSm90Target


def _selected_output_vectorization(module):
    nodes = tuple(
        replace(
            node,
            metadata={
                **dict(node.metadata),
                "selected_vectorization": (
                    "vectorization.matmul.n"
                    if node.id == "projection"
                    else "vectorization.last_axis"
                ),
                "selected_vector_axes": (1,),
                "selected_vector_lanes": ((8,) if node.id == "projection" else (4,)),
            },
        )
        if node.id in {"projection", "logits"}
        else node
        for node in module.nodes
    )
    return replace(module, nodes=nodes)


def _projection_boundary(*, extra_projection_user: bool = False, public_logits: bool = True):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main")

        def forward(self):
            # The selected SM90 pipeline consumes complete BLOCK_K tiles; use
            # a valid physical reduction extent while testing the epilogue
            # fusion rule itself.
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 1024)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (1024, 64)), id="rhs")
            projection = fm.F.math.matmul(lhs, rhs, name="projection")
            logits = fm.F.tensors.cast(projection, fm.DType.FLOAT32, name="logits")
            token = fm.F.nn.greedy_sample(logits, name="token")
            outputs = [token]
            if public_logits:
                outputs.insert(0, logits)
            if extra_projection_user:
                outputs.append(fm.F.math.add(projection, projection, name="extra"))
            self.function("main", (lhs, rhs), tuple(outputs))

    return _selected_output_vectorization(Graph().build())


def _packed_projection_boundary():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 512)), id="lhs")
            weight = self.input(
                "weight", fm.tensor_type("bfloat16", (32, 1024, 2, 64)), id="weight")
            projection = fm.F.math.packed_dense_matmul(
                lhs, weight, name="projection")
            logits = fm.F.tensors.cast(projection, fm.DType.FLOAT32, name="logits")
            token = fm.F.nn.greedy_sample(logits, name="token")
            self.function("main", (lhs, weight), (logits, token))

    return _selected_output_vectorization(Graph().build())


def _mesh_interleaved_packed_projection_boundary():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="distributed", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 512)), id="lhs")
            weight = self.input(
                "weight",
                fm.tensor_type("bfloat16", (32, 149, 128, 2, 64)),
                id="weight",
            )
            projection = fm.F.math.packed_dense_matmul(
                lhs,
                weight,
                packed_layout="k_major_mesh_interleaved_n8_k16",
                logical_n=151936,
                name="projection",
            )
            logits = fm.F.tensors.cast(projection, fm.DType.FLOAT32, name="logits")
            token = fm.F.nn.greedy_sample(logits, name="token")
            self.function("main", (lhs, weight), (logits, token))

    return _selected_output_vectorization(Graph().build())


def test_projection_logits_argmax_match_does_not_fabricate_an_unimplemented_epilogue():
    module = _projection_boundary()
    match = find_projection_logits_argmax_matches(module)["projection"]

    assert match.logits == "logits"
    assert match.sampler == "token"
    assert match.adapters == ()

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(
        point for point in proposed.selection_points
        if point.id == "tir.projection"
    )
    selected = proposed.selection_map[point.id]
    implementation = next(
        candidate for candidate in point.candidates
        if candidate.id == selected.candidate_id
    )
    assert implementation.id == "tir.dense_matmul.gemv"
    assert all(
        candidate.parameters.get("epilogue") != "cast_logits_argmax"
        for candidate in point.candidates
    )


def test_projection_logits_argmax_match_rejects_another_projection_reader():
    assert find_projection_logits_argmax_matches(
        _projection_boundary(extra_projection_user=True)
    ) == {}


def test_projection_logits_argmax_match_requires_public_logits():
    assert find_projection_logits_argmax_matches(
        _projection_boundary(public_logits=False)
    ) == {}


def test_scalar_agent_choice_on_cast_disables_vector_fused_epilogue():
    module = _projection_boundary()
    module = replace(
        module,
        nodes=tuple(
            replace(node, metadata={}) if node.id == "logits" else node
            for node in module.nodes
        ),
    )

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(
        point for point in proposed.selection_points
        if point.id == "tir.projection"
    )
    assert all(
        candidate.parameters.get("epilogue") != "cast_logits_argmax"
        for candidate in point.candidates
    )


def test_packed_large_n_projection_separates_compute_tiles_from_packing():
    module = _packed_projection_boundary()
    match = find_projection_logits_argmax_matches(module)["projection"]
    assert match.logits == "logits"

    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(point for point in proposed.selection_points if point.id == "tir.projection")
    selected = proposed.selection_map[point.id]
    implementation = next(
        candidate for candidate in point.candidates
        if candidate.id == selected.candidate_id
    )

    assert implementation.id == "tir.dense_matmul.packed_k_major_gemv_tn64_bk256"
    assert implementation.parameters["tile_n"] == 64
    assert implementation.parameters["block_k"] == 256
    assert implementation.parameters["packed_layout"] == "k_major_n8_k16"
    assert implementation.parameters.get("epilogue") is None


def test_mesh_interleaved_projection_fails_instead_of_selecting_a_fake_kernel():
    module = _mesh_interleaved_packed_projection_boundary()
    target = NvidiaSm90Target()
    proposed = target.propose_tir(module)

    assert "tir.projection" not in {
        point.id for point in proposed.selection_points
    }
    with pytest.raises(
        IRVerificationError,
        match="No reviewed Triton TIR candidate.*math.packed_dense_matmul",
    ):
        target.lower_to_tir(proposed)
