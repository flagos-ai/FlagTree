# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


class _PartialCombine(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        placement = fm.Placement((3, 5), "ab", "bb")
        value = fm.tensor_type("bfloat16", (1, 16))
        partial_type = fm.DistributedType(
            value,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
            partial=fm.SBP.partial((0, 1)),
        )
        materialized_type = fm.DistributedType(
            value,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        partial = self.input("partial", partial_type, id="partial")
        residual = self.input("residual", materialized_type, id="residual")
        result = fm.F.ntt.matmul_norm_stats_combine(
            partial,
            residual,
            axis=-1,
            use_mean=False,
            name="result",
        )
        self.function("main", (partial, residual), (result,))


class _MaterializedCombine(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        placement = fm.Placement((3, 5), "ab", "bb")
        value_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (1, 16)),
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        projection = self.input("projection", value_type, id="projection")
        residual = self.input("residual", value_type, id="residual")
        result = fm.F.ntt.matmul_norm_stats_combine(
            projection,
            residual,
            axis=-1,
            use_mean=False,
            name="result",
        )
        self.function("main", (projection, residual), (result,))


class _HybridPartialCombine(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        placement = fm.Placement((8, 16), "yx", "bb")
        value = fm.tensor_type("bfloat16", (1, 2048))
        partial_type = fm.DistributedType(
            value,
            (fm.SBP.broadcast(), fm.SBP.split_contiguous((1,), 128)),
            placement,
            partial=fm.SBP.partial((0,)),
        )
        materialized_type = fm.DistributedType(
            value,
            (fm.SBP.broadcast(), fm.SBP.broadcast()),
            placement,
        )
        partial = self.input("partial", partial_type, id="partial")
        residual = self.input("residual", materialized_type, id="residual")
        result = fm.F.ntt.matmul_norm_stats_combine(
            partial,
            residual,
            axis=-1,
            use_mean=False,
            name="result",
        )
        self.function("main", (partial, residual), (result,))


class _TwoAxisShardedMaterializedCombine(fm.Module):
    def __init__(self):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        placement = fm.Placement((8, 16), "yx", "bb")
        value_type = fm.DistributedType(
            fm.tensor_type("bfloat16", (1, 2048)),
            (
                fm.SBP.broadcast(),
                fm.SBP.split_contiguous((0, 1), 16),
            ),
            placement,
        )
        projection = self.input("projection", value_type, id="projection")
        residual = self.input("residual", value_type, id="residual")
        result = fm.F.ntt.matmul_norm_stats_combine(
            projection,
            residual,
            axis=-1,
            use_mean=False,
            name="result",
        )
        self.function("main", (projection, residual), (result,))


class _ShardedPartialCombine(fm.Module):
    """The nncase down-projection gather/reduce distribution contract."""

    def __init__(self):
        super().__init__(dialect="ntt", stage="frozen_constants", entry="main")

    def forward(self):
        placement = fm.Placement((8, 16), "yx", "bb")
        value = fm.tensor_type("bfloat16", (1, 2048))
        partial_type = fm.DistributedType(
            value,
            (
                fm.SBP.broadcast(),
                fm.SBP.split_contiguous((0,), 256),
            ),
            placement,
            partial=fm.SBP.partial((1,)),
        )
        materialized_type = fm.DistributedType(
            value,
            (
                fm.SBP.broadcast(),
                fm.SBP.split_contiguous((0,), 256),
            ),
            placement,
        )
        partial = self.input("partial", partial_type, id="partial")
        residual = self.input("residual", materialized_type, id="residual")
        result = fm.F.ntt.matmul_norm_stats_combine(
            partial,
            residual,
            axis=-1,
            use_mean=False,
            name="result",
        )
        self.function("main", (partial, residual), (result,))


def test_partial_combine_queries_a_target_catalog_without_machine_knobs():
    proposed = NvidiaSm90Target().propose_tir(_PartialCombine().build())
    point = next(
        point for point in proposed.selection_points if point.id == "tir.result"
    )
    candidate = point.candidates[0]

    assert candidate.parameters["family"] == "gather_reduce_add_norm_stats"
    assert candidate.parameters["owner_count"] == 15
    assert candidate.parameters["partial_axes"] == (0, 1)
    assert candidate.parameters["tile"] == 64
    assert candidate.parameters["partial_reduction_width"] == 16
    assert candidate.facts["collective_semantics"] == (
        "gather-reduce-add-norm-stats"
    )
    assert candidate.facts["explicit_norm_stats_result"] is True
    assert tuple(
        workspace["name"] for workspace in candidate.parameters["workspaces"]
    ) == ("collective", "norm_stats_partials")


def test_materialized_combine_can_select_a_single_program_target_implementation():
    proposed = NvidiaSm90Target().propose_tir(_MaterializedCombine().build())
    point = next(
        point for point in proposed.selection_points if point.id == "tir.result"
    )

    assert tuple(candidate.id for candidate in point.candidates) == (
        "tir.add_norm_stats.persistent_rms",
    )
    candidate = point.candidates[0]
    assert candidate.parameters["family"] == "add_norm_stats"
    assert candidate.facts["participant_scope"] == "single_program"
    assert candidate.parameters["owner_count"] == 15
    assert candidate.facts["placement_owner_count"] == 15
    assert candidate.facts["local_semantics"] == "add-norm-stats"
    assert "collective_semantics" not in candidate.facts
    assert "workspaces" not in candidate.parameters


def test_hybrid_partial_combine_uses_partial_axes_not_a_full_mesh_shortcut():
    proposed = NvidiaSm90Target().propose_tir(_HybridPartialCombine().build())
    point = next(
        point for point in proposed.selection_points if point.id == "tir.result"
    )

    assert point.default_candidate == "tir.gather_reduce_add_norm_stats.sum_rms"
    candidate = point.candidates[0]
    assert candidate.parameters["partial_axes"] == (0,)
    assert candidate.parameters["partial_owner_count"] == 8
    assert candidate.parameters["owner_count"] == 128
    assert candidate.facts["placement_owner_count"] == 128


def test_two_axis_sharded_value_selects_owner_local_partial_statistics():
    proposed = NvidiaSm90Target().propose_tir(
        _TwoAxisShardedMaterializedCombine().build()
    )
    point = next(
        point for point in proposed.selection_points if point.id == "tir.result"
    )

    assert tuple(candidate.id for candidate in point.candidates) == (
        "tir.add_norm_stats.local_partial_rms",
    )
    candidate = point.candidates[0]
    assert candidate.parameters["family"] == "add_norm_stats"
    assert candidate.parameters["variant"] == "local_partial_rms"
    assert candidate.parameters["partial_axes"] == ()
    assert candidate.parameters["owner_count"] == 128
    assert candidate.facts["owner_local_stats"] is True
    assert candidate.facts["explicit_norm_stats_result"] is True
    assert "requires" not in candidate.facts


def test_sharded_partial_combine_has_a_reviewed_local_shard_collective():
    proposed = NvidiaSm90Target().propose_tir(_ShardedPartialCombine().build())
    point = next(
        point for point in proposed.selection_points if point.id == "tir.result"
    )

    assert point.default_candidate == (
        "tir.gather_reduce_add_norm_stats.local_shard_sum_rms"
    )
    candidate = point.candidates[0]
    assert candidate.parameters["family"] == "gather_reduce_add_norm_stats"
    assert candidate.parameters["variant"] == "sum_rms"
    assert candidate.parameters["partial_axes"] == (1,)
    assert candidate.parameters["partial_owner_count"] == 16
    assert candidate.parameters["owner_count"] == 128
    assert candidate.facts["distributed_output"] is True
