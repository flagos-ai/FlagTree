# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.targets import NvidiaSm90Target


def _module(*, use_mean=False):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="canonical_constants", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 128)))
            scale = self.input("scale", fm.tensor_type("bfloat16", (128,)))
            bias = fm.F.builtin.splat_const(scale.type, 0.0, name="bias")
            stats = fm.F.nn.norm_stats(
                value, axis=-1, use_mean=use_mean, name="stats")
            output = fm.F.nn.norm_apply(
                value, stats, scale, bias,
                axis=-1, epsilon=1e-6, use_mean=use_mean, name="output",
                metadata={
                    "selected_vectorization": "vectorization.norm_apply.reduction_axis",
                    "selected_vector_axes": (1,),
                    "selected_vector_lanes": (8,),
                },
            )
            self.function("main", (value, scale), (output,))

    return freeze_constant_islands(Graph().build())


def test_replicated_rms_stats_and_apply_use_portable_local_families():
    proposed = NvidiaSm90Target().propose_tir(_module())
    points = {point.id: point for point in proposed.selection_points}

    assert points["tir.stats"].default_candidate == "tir.norm_stats.local"
    assert points["tir.output"].default_candidate == "tir.norm_apply.local"
    assert [value.id for value in points["tir.stats"].candidates] == [
        "tir.norm_stats.local",
    ]
    assert [value.id for value in points["tir.output"].candidates] == [
        "tir.norm_apply.local",
    ]
    apply = next(
        value for value in points["tir.output"].candidates
        if value.id == points["tir.output"].default_candidate)
    assert apply.parameters["family"] == "norm_apply"
    assert apply.parameters["bias_mode"] == "tensor"
    assert apply.parameters["vector_schedule"]["contract"]["kind"] == "reduction_axis"


def test_catalog_does_not_fabricate_unimplemented_layer_norm_tir():
    proposed = NvidiaSm90Target().propose_tir(_module(use_mean=True))
    points = {point.id: point for point in proposed.selection_points}

    assert points["tir.stats"].default_candidate == "tir.norm_stats.local"
    assert points["tir.output"].default_candidate == "tir.norm_apply.local"
    candidate = points["tir.output"].candidates[0]
    assert candidate.parameters["axis"] == 1
    assert candidate.parameters["use_mean"] is True
    assert candidate.parameters["bias_mode"] == "tensor"


def test_non_last_layer_norm_with_nonzero_bias_has_a_generic_apply_candidate():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="canonical_constants", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 3, 5)))
            scale = self.input("scale", fm.tensor_type("bfloat16", (3, 5)))
            bias = self.input("bias", fm.tensor_type("bfloat16", (3, 5)))
            stats = fm.F.nn.norm_stats(value, axis=1, use_mean=True, name="stats")
            output = fm.F.nn.norm_apply(
                value,
                stats,
                scale,
                bias,
                axis=1,
                epsilon=1e-5,
                use_mean=True,
                name="output",
            )
            self.function("main", (value, scale, bias), (output,))

    proposed = NvidiaSm90Target().propose_tir(Graph().build())
    point = next(value for value in proposed.selection_points if value.id == "tir.output")

    assert point.default_candidate == "tir.norm_apply.local"
    assert [candidate.id for candidate in point.candidates] == ["tir.norm_apply.local"]
    assert point.candidates[0].parameters["axis"] == 1
    assert point.candidates[0].parameters["use_mean"] is True
    assert point.candidates[0].parameters["bias_mode"] == "tensor"


def test_non_last_axis_stats_use_the_same_generic_local_implementation():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="nn", stage="canonical_constants", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 3, 5)))
            stats = fm.F.nn.norm_stats(
                value, axis=1, use_mean=True, name="stats"
            )
            self.function("main", (value,), (stats,))

    proposed = NvidiaSm90Target().propose_tir(Graph().build())
    point = next(value for value in proposed.selection_points if value.id == "tir.stats")

    assert point.default_candidate == "tir.norm_stats.local"
    candidate = point.candidates[0]
    assert candidate.parameters["axis"] == 1
    assert candidate.parameters["use_mean"] is True
    assert candidate.facts["additive_statistics"] == ("sum", "square_sum")
