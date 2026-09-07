# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse a private partial-statistics boxing into normalization apply."""

from __future__ import annotations

from dataclasses import dataclass, replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.ops.ntt.gather_reduce_norm_apply import (
    GatherReduceNormApply,
)


_SPLAT_PRESERVING_VIEWS = frozenset({
    "builtin.identity",
    "distributed.sharded_view",
    "tensors.bitcast",
    "tensors.pack",
    "tensors.reshape",
    "tensors.unpack",
})


def fuse_gather_reduce_norm_apply(module: IRModule) -> IRModule:
    """Remove single-use Sum-partial materialization before ``NormApply``.

    The proof is over SSA uses before bufferization.  At this point every
    ``distributed.boxing`` still denotes one exact value region, so a single
    use is equivalent to nncase's exact-buffer-region use-count requirement
    without depending on allocation identities that do not exist yet.
    """

    users = _users(module)
    removed: set[str] = set()
    replacements: dict[str, Node] = {}
    for norm in module.nodes:
        if norm.op != "nn.norm_apply" or len(norm.inputs) != 4:
            continue
        boxing = module.node_map.get(norm.inputs[1])
        if (
            boxing is None
            or boxing.op != "distributed.boxing"
            or len(boxing.inputs) != 1
            or users.get(boxing.id) != (norm.id,)
        ):
            continue
        partial = module.node_map.get(boxing.inputs[0])
        if partial is None:
            continue
        attrs = {
            "materialized_stats_type": boxing.type,
            "axis": int(norm.attrs["axis"]),
            "epsilon": float(norm.attrs["epsilon"]),
            "use_mean": bool(norm.attrs["use_mean"]),
            "round_before_scale": bool(norm.attrs.get("round_before_scale", False)),
            "has_bias": not _is_zero_splat(
                module.node_map[norm.inputs[3]], module
            ),
        }
        inputs = (
            partial,
            module.node_map[norm.inputs[0]],
            module.node_map[norm.inputs[2]],
            module.node_map[norm.inputs[3]],
        )
        try:
            result_type = GatherReduceNormApply.infer_call_type(inputs, attrs)
            effect = GatherReduceNormApply.infer_effect(inputs, attrs)
        except (IRSchemaError, TypeError, ValueError, KeyError):
            continue
        if result_type != norm.type:
            continue
        replacements[norm.id] = replace(
            norm,
            op=GatherReduceNormApply.op_name,
            inputs=tuple(value.id for value in inputs),
            type=result_type,
            effect=effect,
            attrs=attrs,
            metadata={
                **dict(norm.metadata),
                "fused_partial_stats_materialization": boxing.id,
                "partial_stats_source": partial.id,
            },
        )
        removed.add(boxing.id)
    if not replacements:
        return module

    removed_points = {
        point.id for point in module.selection_points if point.owner in removed
    }
    return replace(
        module,
        nodes=tuple(
            replacements.get(node.id, node)
            for node in module.nodes
            if node.id not in removed
        ),
        selection_points=tuple(
            point
            for point in module.selection_points
            if point.id not in removed_points
        ),
        selections=tuple(
            record
            for record in module.selections
            if record.point_id not in removed_points
        ),
    )


def _users(module: IRModule) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            result.setdefault(input_id, []).append(node.id)
    for function in module.functions:
        for output in function.outputs:
            result.setdefault(output, []).append(f"@{function.name}:return")
    return {key: tuple(value) for key, value in result.items()}


def _is_zero_splat(node: Node, module: IRModule) -> bool:
    observed: set[str] = set()
    while (
        node.op in _SPLAT_PRESERVING_VIEWS
        and len(node.inputs) == 1
        and node.id not in observed
    ):
        observed.add(node.id)
        node = module.node_map[node.inputs[0]]
    if node.op == "builtin.splat_const":
        return float(node.attrs["value"]) == 0.0
    if node.op != "builtin.const_asset":
        return False
    recipe = next(
        (
            value
            for value in module.constant_recipes
            if value.id == str(node.attrs["recipe"])
        ),
        None,
    )
    if recipe is None:
        return False
    recipe_nodes = recipe.node_map
    source = recipe_nodes.get(str(node.attrs["output"]))
    observed.clear()
    while (
        source is not None
        and source.op in _SPLAT_PRESERVING_VIEWS
        and len(source.inputs) == 1
        and source.id not in observed
    ):
        observed.add(source.id)
        source = recipe_nodes.get(source.inputs[0])
    return (
        source is not None
        and source.op == "builtin.splat_const"
        and float(source.attrs["value"]) == 0.0
    )


@dataclass(frozen=True)
class FuseGatherReduceNormApplyPass:
    name: str = "FuseGatherReduceNormApply"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        return fuse_gather_reduce_norm_apply(module)


__all__ = [
    "FuseGatherReduceNormApplyPass",
    "fuse_gather_reduce_norm_apply",
]
