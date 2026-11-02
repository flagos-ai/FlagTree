# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""RDNA feeds NormStats its 16-bit residual instead of the exact f32 widening, so the residual
add + statistics formation applies; other targets keep the cast."""

from triton.flagmega import ir as fm
from triton.flagmega.passes.functions.form_add_norm_stats import form_add_norm_stats
from triton.flagmega.passes.norm_stats import fold_widening_norm_stats_casts
from triton.flagmega.targets import get_target


class _ResidualNorm(fm.Module):
    def __init__(self, shared_cast=False):
        super().__init__(dialect="nn", stage="decomposed", entry="main")
        self.shared_cast = shared_cast

    def forward(self):
        hidden = self.input("hidden", fm.tensor_type("bfloat16", [1, 64]))
        delta = self.input("delta", fm.tensor_type("bfloat16", [1, 64]))
        weight = self.input("weight", fm.tensor_type("float32", [64]))
        bias = fm.F.builtin.splat_const(weight.type, 0.0, name="bias")
        residual = fm.F.math.add(hidden, delta, name="residual")
        widened = fm.F.tensors.cast(residual, "float32", name="widened")
        stats = fm.F.nn.norm_stats(widened, axis=-1, use_mean=False, name="stats")
        output = fm.F.nn.norm_apply(residual, stats, weight, bias, axis=-1, epsilon=1e-6, use_mean=False,
                                    output_dtype="bfloat16", name="output")
        outputs = (output, residual, widened) if self.shared_cast else (output, residual)
        self.function("main", (hidden, delta, weight), outputs)


def test_fold_lets_the_residual_add_and_statistics_form_one_combine():
    module = _ResidualNorm().build()
    assert not any(node.op == "ntt.add_norm_stats" for node in form_add_norm_stats(module).nodes)
    folded = fold_widening_norm_stats_casts(module)
    assert "widened" not in folded.node_map
    assert folded.node_map["stats"].inputs[0] == "residual"
    assert any(node.op == "ntt.add_norm_stats" for node in form_add_norm_stats(folded).nodes)


def test_fold_keeps_a_cast_with_other_users():
    folded = fold_widening_norm_stats_casts(_ResidualNorm(shared_cast=True).build())
    assert folded.node_map["stats"].inputs[0] == "residual"
    assert folded.node_map["widened"].op == "tensors.cast"


def test_only_rdna_rewrites_before_vectorization():
    for name in ("amd-gfx1100", "amd-gfx1201"):
        assert getattr(get_target(name).machine, "pre_vectorization_rewrite", None) is not None
    machine = getattr(get_target("nvidia-sm120"), "machine", None)
    assert getattr(machine, "pre_vectorization_rewrite", None) is None
