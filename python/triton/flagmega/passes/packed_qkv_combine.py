# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-aligned post-distribution packed Q/K/V combine passes."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import IRModule
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.ntt import (
    fold_materialized_packed_qkv_parallel_linear_combine_rule,
    lower_packed_qkv_parallel_linear_combine_rule,
)


def fold_materialized_packed_qkv_parallel_linear_combine(
    module: IRModule,
) -> IRModule:
    """Remove exact materialized identity combines after layout extraction."""

    removed_points = {
        point.id
        for point in module.selection_points
        if point.owner is not None
        and module.node_map[point.owner].op
        == "ntt.packed_qkv_parallel_linear_combine"
    }
    # A redirect normally transfers selection-point ownership to its target.
    # Here the source point describes the combine relation while the target
    # already owns a different projection candidate.  Remove the obsolete
    # point before the rewriter's mandatory per-step verification so it can
    # never be rebound to the projection under the combine record.
    prepared = replace(
        module,
        selection_points=tuple(
            point for point in module.selection_points if point.id not in removed_points
        ),
        selections=tuple(
            selection
            for selection in module.selections
            if selection.point_id not in removed_points
        ),
    )
    result = DataflowPass(
        "FoldMaterializedPackedQKVParallelLinearCombine",
        (fold_materialized_packed_qkv_parallel_linear_combine_rule(),),
    ).run(prepared)
    return result


def lower_packed_qkv_parallel_linear_combine(module: IRModule) -> IRModule:
    """Lower an unfused partial combine to generic distributed Boxing."""

    return DataflowPass(
        "LowerPackedQKVParallelLinearCombine",
        (lower_packed_qkv_parallel_linear_combine_rule(),),
    ).run(module)


__all__ = [
    "fold_materialized_packed_qkv_parallel_linear_combine",
    "lower_packed_qkv_parallel_linear_combine",
]
