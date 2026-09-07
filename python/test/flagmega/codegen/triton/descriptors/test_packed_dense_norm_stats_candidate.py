# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Candidate legality for packed projection/residual/RMS-statistics fusion."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.targets import NvidiaSm90Target


_IMPLEMENTATION = (
    "tir.dense_matmul."
    "packed_tensor_descriptor_smem_pipeline_gemv_norm_stats"
)
_TABLE_IMPLEMENTATION = (
    "tir.dense_matmul."
    "packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats"
)
_PORTABLE_IMPLEMENTATION = (
    "tir.dense_matmul.packed_k_major_gemv_norm_stats"
)
_STAGED_IMPLEMENTATION = (
    "tir.dense_matmul."
    "packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192"
)


def _module(
    *,
    rows: int = 1,
    reduction_extent: int = 2048,
    output_extent: int = 2048,
    output_mesh_axes: tuple[int, ...] = (0, 1),
    output_block: int | None = None,
) -> fm.IRModule:
    if reduction_extent % 16 or output_extent % 8:
        raise ValueError("Test shapes must preserve the K16/N8 packing atoms.")
    placement = fm.Placement((8, 16), "yx", "bb")
    broadcast = fm.SBP.broadcast()
    output_policy = (
        fm.SBP.split_contiguous(output_mesh_axes)
        if output_mesh_axes
        else broadcast
    )
    if output_block is not None:
        output_policy = fm.SBP.split_block_cyclic(output_mesh_axes, output_block)
    lhs_type = fm.DistributedType(
        fm.tensor_type("bfloat16", (rows, reduction_extent)),
        (broadcast, broadcast),
        placement,
    )
    rhs_type = fm.DistributedType(
        fm.tensor_type(
            fm.vector_type("bfloat16", (8, 2, 8)),
            (reduction_extent // 16, output_extent // 8),
        ),
        (broadcast, output_policy),
        placement,
    )
    value_type = fm.DistributedType(
        fm.tensor_type(
            fm.vector_type("bfloat16", (8,)),
            (rows, output_extent // 8),
        ),
        (broadcast, output_policy),
        placement,
    )
    stats_type = fm.DistributedType(
        fm.tensor_type("float32", (1, rows, 1)),
        (broadcast, broadcast, broadcast),
        placement,
        partial=(fm.SBP.partial(output_mesh_axes) if output_mesh_axes else None),
    )
    lhs = fm.Node("lhs", "builtin.var", (), lhs_type)
    rhs = fm.Node("rhs", "builtin.var", (), rhs_type)
    addend = fm.Node("addend", "builtin.var", (), value_type)
    residual_add = fm.Node(
        "residual_add",
        "math.add",
        (addend.id, addend.id),
        value_type,
        metadata={
            "selected_vectorization": "vectorization.last_axis",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    norm_consumer = fm.Node(
        "norm_consumer",
        "nn.norm_apply",
        (addend.id, addend.id, addend.id, addend.id),
        value_type,
        attrs={"axis": 1, "epsilon": 1e-6, "use_mean": False},
        metadata={
            "selected_vectorization": "vectorization.norm_apply.reduction_axis",
            "selected_vector_axes": (1,),
            "selected_vector_lanes": (8,),
        },
    )
    result = fm.Node(
        "result",
        "ntt.matmul_norm_stats",
        (lhs.id, rhs.id, addend.id),
        fm.TupleType((value_type, stats_type)),
        attrs={
            "transpose_a": False,
            "transpose_b": False,
            "rhs_layout": "k_major",
            "axis": 1,
            "use_mean": False,
        },
        metadata={
            "matmul_vectorization": {
                "selected_vectorization": "vectorization.matmul.n",
                "selected_vector_axes": (1,),
                "selected_vector_lanes": (8,),
            },
            "residual_add": residual_add.id,
            "residual_input": addend.id,
            "norm_consumer": norm_consumer.id,
            "projection_adapters": ("projection_view",),
        },
    )
    return fm.IRModule(
        dialect="distributed",
        stage="matmul_norm_stats_lowered",
        nodes=(lhs, rhs, addend, residual_add, norm_consumer, result),
        functions=(),
        entry="main",
    )


def _candidate_ids(module: fm.IRModule) -> set[str]:
    proposed = NvidiaSm90Target().propose_tir(module)
    point = next(
        (
            value
            for value in proposed.selection_points
            if value.id == "tir.result"
        ),
        None,
    )
    return set() if point is None else {candidate.id for candidate in point.candidates}


def test_packed_norm_stats_owner_table_is_the_reviewed_default_for_full_mesh_n_tiles():
    proposed = NvidiaSm90Target().propose_tir(_module())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == _TABLE_IMPLEMENTATION
    candidate = next(
        value for value in point.candidates if value.id == _TABLE_IMPLEMENTATION
    )
    assert candidate.parameters["packed_layout"] == "k_major_n8_k16"
    assert candidate.parameters["epilogue"] == "residual_norm_stats"
    assert candidate.parameters["owner_count"] == 128
    assert candidate.parameters["explicit_results"] == ("value", "norm_stats")
    assert candidate.facts["explicit_norm_stats_result"] is True
    assert candidate.facts["materializes_residual"] is True
    assert candidate.facts["owner_local_stats"] is True
    assert candidate.facts["internal_grid_barriers"] == 0
    assert candidate.facts["host_tensor_descriptor_table"] is True


def test_packed_norm_stats_owner_descriptor_table_is_a_typed_candidate():
    proposed = NvidiaSm90Target().propose_tir(_module())
    point = next(value for value in proposed.selection_points if value.id == "tir.result")
    candidate = next(
        value for value in point.candidates if value.id == _TABLE_IMPLEMENTATION
    )

    assert point.default_candidate == _TABLE_IMPLEMENTATION
    assert candidate.parameters["descriptor_kind"] == "table"
    assert candidate.facts["host_tensor_descriptor_table"] is True
    assert candidate.facts["owner_local_stats"] is True


@pytest.mark.parametrize("extent", [2048, 4096, 8192])
def test_table_supports_affine_cyclic_n_without_enabling_global_descriptor(extent):
    ids = _candidate_ids(_module(output_extent=extent, output_block=1))
    assert _TABLE_IMPLEMENTATION in ids
    assert _STAGED_IMPLEMENTATION in ids
    assert _IMPLEMENTATION not in ids


def test_table_rejects_non_affine_cyclic_n_but_keeps_local_simt_candidate():
    ids = _candidate_ids(_module(output_extent=8192, output_block=2))
    assert _TABLE_IMPLEMENTATION not in ids
    assert _PORTABLE_IMPLEMENTATION in ids


def test_packed_norm_stats_pipeline_rejects_an_incomplete_k_tile():
    assert _IMPLEMENTATION not in _candidate_ids(
        _module(reduction_extent=1536)
    )


def test_packed_norm_stats_pipeline_requires_every_placement_owner_to_own_n():
    assert _IMPLEMENTATION not in _candidate_ids(
        _module(output_mesh_axes=(0,))
    )


def test_packed_norm_stats_has_portable_local_shard_candidate_for_broadcast_output():
    proposed = NvidiaSm90Target().propose_tir(_module(output_mesh_axes=()))
    point = next(value for value in proposed.selection_points if value.id == "tir.result")

    assert point.default_candidate == _PORTABLE_IMPLEMENTATION
    candidate = next(
        value for value in point.candidates if value.id == _PORTABLE_IMPLEMENTATION
    )
    assert candidate.parameters["statistics_kind"] == "local_shard"
    assert candidate.parameters["output_partition_axes"] == ()
    assert candidate.facts["portable_triton"] is True


def test_packed_norm_stats_pipeline_rejects_multiple_outer_rows():
    assert _IMPLEMENTATION not in _candidate_ids(_module(rows=2))


@pytest.mark.parametrize("reduction_extent", [1024, 2048, 4096, 6144, 8192])
def test_packed_norm_stats_lhs_staging_is_capacity_bounded_not_shape_specific(reduction_extent):
    proposed = NvidiaSm90Target().propose_tir(_module(reduction_extent=reduction_extent))
    point = next(value for value in proposed.selection_points if value.id == "tir.result")
    candidate = next((value for value in point.candidates if value.id == _STAGED_IMPLEMENTATION), None)
    assert candidate is not None
    assert candidate.parameters["lhs_stage_extent"] == 8192
    assert candidate.parameters["lhs_copy_tile"] == 1024
    # The measured default is unchanged until an independent performance review.
    assert point.default_candidate == _TABLE_IMPLEMENTATION


@pytest.mark.parametrize("reduction_extent", [1536, 9216, 16384])
def test_packed_norm_stats_lhs_staging_rejects_unsupported_capacity_or_transfer_tile(reduction_extent):
    assert _STAGED_IMPLEMENTATION not in _candidate_ids(_module(reduction_extent=reduction_extent))
