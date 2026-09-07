# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fuse explicit RMS statistics/apply when the selected ABI is replicated."""

from __future__ import annotations

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, IRModule, Node, TensorType, get_definition
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import RewriteRule
from triton.flagmega.rules.neutral._utility import decomposition_metadata


_TRANSPARENT = frozenset({"distributed.boxing", "distributed.sharded_view"})


def fuse_norm_stats_apply_rule() -> RewriteRule:
    pattern = F.nn.is_norm_apply(
        wildcard("input", type_pattern=NormApply.value.type_pattern),
        wildcard("stats", type_pattern=NormApply.stats.type_pattern),
        wildcard("scale", type_pattern=NormApply.scale.type_pattern),
        wildcard("bias", type_pattern=NormApply.bias.type_pattern),
        target_name="target",
        call_name="call",
    )

    def rewrite(result, module: IRModule):
        source = result["call"]
        value = result["input"]
        stats = result["stats"]
        scale = result["scale"]
        bias = result["bias"]
        assert all(isinstance(item, Node) for item in (source, value, stats, scale, bias))
        if bool(source.attrs["use_mean"]):
            return source
        if not bool(source.attrs.get("round_before_scale", False)):
            return source
        value_tensor = tensor_of(value.type)
        axis = int(source.attrs["axis"])
        normalized_axis = axis + value_tensor.rank if axis < 0 else axis
        if normalized_axis != value_tensor.rank - 1:
            return source
        stats_source = _strip_adapters(stats, module)
        if (
            stats_source.op != "nn.norm_stats"
            or bool(stats_source.attrs["use_mean"])
            or _normalized_axis(stats_source, module) != normalized_axis
        ):
            return source
        stats_input = _strip_adapters(module.node_map[stats_source.inputs[0]], module)
        value_input = _strip_adapters(value, module)
        if stats_input.id != value_input.id:
            return source
        bias_source = _strip_adapters(bias, module)
        if bias_source.op != "builtin.splat_const" or float(bias_source.attrs["value"]) != 0.0:
            return source
        if isinstance(value.type, DistributedType) and not all_broadcast(value.type):
            return source
        if isinstance(scale.type, DistributedType) and not all_broadcast(scale.type):
            return source
        definition = get_definition("nn.rms_norm")
        try:
            prepared = definition.prepare(
                (value, scale),
                {"epsilon": float(source.attrs["epsilon"]), "weight_bias": 0.0},
            )
        except IRSchemaError:
            return source
        if prepared.result_type != source.type:
            return source
        return Node(
            source.id,
            "nn.rms_norm",
            tuple(item.id for item in prepared.inputs),
            prepared.result_type,
            prepared.effect,
            prepared.attrs,
            {
                **decomposition_metadata(source, "FuseNormStatsApply"),
                "fused_norm_stats": stats_source.id,
                "fused_bias": bias_source.id,
            },
        )

    return RewriteRule("FuseNormStatsApply", pattern, rewrite)


def _strip_adapters(node: Node, module: IRModule) -> Node:
    observed: set[str] = set()
    while node.op in _TRANSPARENT and len(node.inputs) == 1:
        if node.id in observed:
            raise IRSchemaError("Normalization adapter chain contains a cycle.")
        observed.add(node.id)
        node = module.node_map[node.inputs[0]]
    return node


def _normalized_axis(node: Node, module: IRModule) -> int:
    value = tensor_of(module.node_map[node.inputs[0]].type)
    axis = int(node.attrs["axis"])
    return axis + value.rank if axis < 0 else axis


__all__ = ["fuse_norm_stats_apply_rule"]
