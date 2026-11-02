# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Post-distribution normalization graph transformations."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.ir import DType, IRModule, TensorType, verify_module
from triton.flagmega.passes.functions.equivalent_variants import merge_equivalent_function_variants
from triton.flagmega.passes.rewriter import DataflowPass
from triton.flagmega.rules.neutral import (
    fold_bind_norm_stats_rule,
    fuse_norm_stats_apply_rule,
)
from triton.flagmega.rules.ntt import lower_add_norm_stats_rule


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


def lower_add_norm_stats(module: IRModule) -> IRModule:
    """Fuse a private logical MatMul into its explicit value/stats combine."""

    return DataflowPass(
        "LowerAddNormStats",
        (lower_add_norm_stats_rule(),),
    ).run(module)


def fold_widening_norm_stats_casts(module: IRModule) -> IRModule:
    """Feed ``NormStats`` the 16-bit value instead of its exact f32 widening.

    NormStats squares and sums in f32 whatever its input dtype, so the cast is
    redundant; without it the residual-add/statistics formation and the
    cross-function threading see the add or parameter they match. A cast with
    other users stays in place for them.
    """

    node_map = module.node_map
    users: dict[str, int] = {}
    for node in module.nodes:
        for value in node.inputs:
            users[value] = users.get(value, 0) + 1
    outputs = {value for function in module.functions for value in function.outputs}
    folded, nodes = set(), []
    for node in module.nodes:
        if node.op == "nn.norm_stats":
            cast = node_map[node.inputs[0]]
            source = node_map[cast.inputs[0]] if cast.op == "tensors.cast" else None
            if (
                source is not None
                and isinstance(cast.type, TensorType)
                and cast.type.dtype == DType.FLOAT32
                and isinstance(source.type, TensorType)
                and source.type.dtype in {DType.BFLOAT16, DType.FLOAT16}
            ):
                node = replace(node, inputs=(source.id, *node.inputs[1:]))
                if users[cast.id] == 1 and cast.id not in outputs:
                    folded.add(cast.id)
        nodes.append(node)
    if not folded and all(a is b for a, b in zip(nodes, module.nodes)):
        return module
    return verify_module(replace(module, nodes=tuple(node for node in nodes if node.id not in folded)))


__all__ = [
    "finalize_norm_stats_bindings",
    "fold_widening_norm_stats_casts",
    "fuse_norm_stats_apply",
    "lower_add_norm_stats",
]
