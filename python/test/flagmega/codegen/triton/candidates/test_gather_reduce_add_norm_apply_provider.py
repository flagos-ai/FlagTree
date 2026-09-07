# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


class _Graph(fm.Module):
    def __init__(self, *, use_mean=False):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")
        self.use_mean = use_mean

    def forward(self):
        placement = fm.Placement((3, 5), "ab", "bb")
        value_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (1, 16)),
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        partial_type = fm.DistributedType(
            value_type.tensor,
            value_type.axis_policies,
            placement,
            fm.SBP.partial((0, 1)),
        )
        parameter_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (16,)),
            (fm.SBP.broadcast(),),
            placement,
        )
        partial = self.input("partial", partial_type, id="partial")
        residual = self.input("residual", value_type, id="residual")
        scale = self.input("scale", parameter_type, id="scale")
        bias = self.input("bias", parameter_type, id="bias")
        result = fm.F.ntt.gather_reduce_add_norm_apply(
            partial,
            residual,
            scale,
            bias,
            axis=-1,
            epsilon=1e-6,
            use_mean=self.use_mean,
            name="result",
        )
        self.function("main", (partial, residual, scale, bias), (result,))


def test_provider_uses_portable_collective_contract_and_private_workspace():
    proposed = NvidiaSm90Target().propose_tir(_Graph().build())
    point = next(point for point in proposed.selection_points if point.id == "tir.result")
    candidate = point.candidates[0]

    assert point.default_candidate == "tir.gather_reduce_add_norm_apply.sum"
    assert candidate.parameters["family"] == "gather_reduce_add_norm_apply"
    assert candidate.parameters["partial_axes"] == (0, 1)
    assert candidate.parameters["partial_owner_count"] == 15
    assert candidate.parameters["owner_count"] == 15
    assert candidate.facts["internal_grid_barriers"] == 1
    assert candidate.facts["private_norm_stats_workspace"] is True
    assert tuple(
        workspace["name"] for workspace in candidate.parameters["workspaces"]
    ) == ("norm_stats_partials",)
    assert candidate.parameters["workspaces"][0]["type"].shape == (
        fm.dim(1), fm.dim(15)
    )


def test_mean_variant_allocates_two_statistics_components():
    proposed = NvidiaSm90Target().propose_tir(_Graph(use_mean=True).build())
    point = next(point for point in proposed.selection_points if point.id == "tir.result")

    assert point.candidates[0].parameters["workspaces"][0]["type"].shape == (
        fm.dim(2), fm.dim(15)
    )
