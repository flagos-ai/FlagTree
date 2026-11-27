# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit late formation of packed residual/local-statistics pairs."""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Node, PURE, verify_module
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.ntt.vectorized_add_norm_stats import VectorizedAddNormStats, LOCAL16_ORDER


def form_vectorized_add_norm_stats(module, *, pairs):
    """Keep both original result identities and all users, including exports.

    Selection is explicit: this helper does not alter default formation or
    turn an ordered complete-vector NormStats into an owner-local reduction.
    """
    verify_module(module)
    nodes = module.node_map
    changes, prefixes, occupied = {}, {}, set(nodes)
    for raw_id, stats_id in pairs:
        raw, stats = nodes[raw_id], nodes[stats_id]
        if (raw.op != "math.vectorized_binary" or raw.attrs.get("binary_op") != "add" or not raw.effect.is_pure
                or raw.attrs.get("post_ops") or stats.op != "nn.norm_stats" or stats.inputs != (raw.id, )
                or not stats.effect.is_pure or has_ops(stats.attrs) or stats.attrs.get("use_mean") is not False
                or stats.attrs.get("reduction_order", "unspecified") != "unspecified"
                or stats.attrs.get("reduction_group_size", 1) != 1):
            raise IRSchemaError("Vectorized add/statistics formation requires the explicit pure unordered local pair.")
        if raw_id in changes or stats_id in changes:
            raise IRSchemaError("Vectorized add/statistics formation pairs overlap.")
        identifier = f"{raw_id}.local_stats"
        if identifier in occupied:
            raise IRSchemaError("Vectorized add/statistics combined identity already exists.")
        attrs = {**raw.attrs, "axis": stats.attrs["axis"], "use_mean": False, "local_reduction_order": LOCAL16_ORDER}
        prepared = VectorizedAddNormStats.prepare(tuple(nodes[k] for k in raw.inputs), attrs)
        if prepared.result_type.fields != (raw.type, stats.type):
            raise IRSchemaError("Vectorized add/statistics formation changed raw/P types or ownership.")
        combined = Node(identifier, VectorizedAddNormStats.op_name, raw.inputs, prepared.result_type, prepared.effect,
                        prepared.attrs, {
                            **raw.metadata, "introduced_by": "FormVectorizedAddNormStats", "raw_result": raw_id,
                            "stats_result": stats_id
                        })
        prefixes[raw_id] = combined
        for index, old in enumerate((raw, stats)):
            changes[old.id] = Node(old.id, "builtin.get_item", (identifier, ), old.type, PURE, {"index": index},
                                   {**old.metadata, "introduced_by": "FormVectorizedAddNormStats"})
        occupied.add(identifier)
    result = []
    for node in module.nodes:
        if node.id in prefixes:
            result.append(prefixes[node.id])
        result.append(changes.get(node.id, node))
    points = tuple(p for p in module.selection_points if p.owner not in changes)
    ids = {p.id for p in points}
    return verify_module(
        replace(module, nodes=tuple(result), selection_points=points,
                selections=tuple(s for s in module.selections if s.point_id in ids)))


__all__ = ["form_vectorized_add_norm_stats"]
