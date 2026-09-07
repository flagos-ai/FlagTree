# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import (
    DistributedCandidateContext,
    NormApplyCandidateProvider,
    NormStatsCandidateProvider,
)


class _Module(fm.Module):
    def __init__(self, stage="packed"):
        super().__init__(dialect="nn", stage=stage, entry="main")

    def forward(self):
        value = self.input("value", fm.tensor_type("bfloat16", [1, 128]))
        scale = self.input("scale", fm.tensor_type("bfloat16", [128]))
        bias = self.input("bias", fm.tensor_type("bfloat16", [128]))
        stats = fm.F.nn.norm_stats(value, axis=-1, use_mean=False, name="stats")
        output = fm.F.nn.norm_apply(
            value,
            stats,
            scale,
            bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=False,
            name="output",
        )
        self.function("main", (value, scale, bias), (output,))


def test_norm_stats_provider_exposes_partial_for_every_legal_mesh_reduction_split():
    module = _Module().build()
    placement = fm.Placement((2, 4), "yx", "bb")
    context = DistributedCandidateContext(
        module, module.node_map["stats"], placement, ((),))
    candidates = NormStatsCandidateProvider().get_candidates(context)

    partial_axes = {
        candidate.return_type.partial.axes
        for candidate in candidates
        if isinstance(candidate.return_type, fm.DistributedType)
        and candidate.return_type.partial is not None
    }
    assert partial_axes == {(0,), (1,), (0, 1)}
    assert all(
        candidate.objective_model == "flagmega.target-op-cost.unit/v1"
        for candidate in candidates
    )


def test_norm_apply_provider_requires_materialized_stats_and_matching_suffix_policy():
    module = _Module().build()
    placement = fm.Placement((2, 4), "yx", "bb")
    context = DistributedCandidateContext(
        module, module.node_map["output"], placement, ((), (), (), ()))
    candidates = NormApplyCandidateProvider().get_candidates(context)
    sharded = next(
        candidate for candidate in candidates
        if isinstance(candidate.return_type, fm.DistributedType)
        and hasattr(candidate.return_type.axis_policies[-1], "hierarchy_axes")
        and candidate.return_type.axis_policies[-1].hierarchy_axes == (0, 1)
    )

    value_type, stats_type, scale_type, bias_type = sharded.input_types
    assert isinstance(value_type, fm.DistributedType)
    assert isinstance(stats_type, fm.DistributedType) and stats_type.partial is None
    assert stats_type.axis_policies == (
        fm.SBP.broadcast(), fm.SBP.broadcast(), fm.SBP.broadcast())
    assert scale_type.axis_policies == (value_type.axis_policies[-1],)
    assert bias_type.axis_policies == (value_type.axis_policies[-1],)


def test_norm_providers_preserve_an_available_noncanonical_split_granularity():
    module = _Module().build()
    placement = fm.Placement((8, 16), "yx", "bb")
    value_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [1, 128]),
        (fm.SBP.broadcast(), fm.SBP.split_contiguous((0,), 32)),
        placement,
    )
    stats_context = DistributedCandidateContext(
        module,
        module.node_map["stats"],
        placement,
        ((value_type,),),
    )

    stats_candidates = NormStatsCandidateProvider().get_candidates(stats_context)

    preserved_stats = next(
        candidate
        for candidate in stats_candidates
        if candidate.input_types == (value_type,)
    )
    assert preserved_stats.return_type.partial == fm.SBP.partial((0,))

    materialized_stats = fm.DistributedType(
        preserved_stats.return_type.tensor,
        preserved_stats.return_type.axis_policies,
        placement,
    )
    scale_type = fm.DistributedType(
        fm.tensor_type("bfloat16", [128]),
        (fm.SBP.split_contiguous((0,), 32),),
        placement,
    )
    apply_context = DistributedCandidateContext(
        module,
        module.node_map["output"],
        placement,
        ((value_type,), (materialized_stats,), (scale_type,), (scale_type,)),
    )

    apply_candidates = NormApplyCandidateProvider().get_candidates(apply_context)

    assert any(candidate.return_type == value_type for candidate in apply_candidates)
