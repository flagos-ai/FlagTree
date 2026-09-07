# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-distribution normalization graph transformations."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import IRModule
from triton.flagmega.passes.functions.equivalent_variants import merge_equivalent_function_variants
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import (
    fold_bind_norm_stats_rule,
    fuse_norm_stats_apply_rule,
)
from triton.flagmega.rules.ntt import lower_matmul_norm_stats_combine_rule


def finalize_norm_stats_bindings(module: IRModule) -> IRModule:
    """Finalize bindings and reusable variants after layouts reach a fixed point."""

    removed_points = {
        point.id
        for point in module.selection_points
        if point.owner is not None
        and module.node_map[point.owner].op == "nn.bind_norm_stats"
    }
    result = DataflowPass(
        "FoldBindNormStats", (fold_bind_norm_stats_rule(),)).run(module)
    if removed_points:
        result = replace(
            result,
            selection_points=tuple(
                value for value in result.selection_points
                if value.id not in removed_points
            ),
            selections=tuple(
                value for value in result.selections
                if value.point_id not in removed_points
            ),
        )
    return merge_equivalent_function_variants(result)


def fuse_norm_stats_apply(module: IRModule) -> IRModule:
    """Select the existing RMSNorm kernel ABI for a replicated explicit pair."""

    return DataflowPass(
        "FuseNormStatsApply", (fuse_norm_stats_apply_rule(),)).run(module)


def lower_matmul_norm_stats_combine(module: IRModule) -> IRModule:
    """Fuse a private logical MatMul into its explicit value/stats combine."""

    return DataflowPass(
        "LowerMatMulNormStatsCombine",
        (lower_matmul_norm_stats_combine_rule(),),
    ).run(module)


__all__ = [
    "finalize_norm_stats_bindings",
    "fuse_norm_stats_apply",
    "lower_matmul_norm_stats_combine",
]
