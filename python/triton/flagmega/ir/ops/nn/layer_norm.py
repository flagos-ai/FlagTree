# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fused LayerNorm semantic op, decomposed before optimization like nncase."""

from __future__ import annotations

from typing import Mapping, Sequence

from triton.flagmega.ir.model import IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn._norm import norm_apply_value, norm_stats_value
from triton.flagmega.ir.ops.nn.norm_apply import NormApply
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "nn.layer_norm",
    namespace="nn",
    functional_name="layer_norm",
    display_name="NN.LayerNorm",
)
class LayerNorm(OpDefinition):
    value = input_parameter(is_tensor(), name="input")
    scale = input_parameter(is_tensor())
    bias = input_parameter(is_tensor())
    axis = attribute_parameter()
    epsilon = attribute_parameter()
    use_mean = attribute_parameter(default=True)
    round_before_scale = attribute_parameter(default=False)

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        return NormApply.normalize_attrs(attributes)

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        stats_type = NormStats.infer_type((cls.value.read(inputs),), {
            "axis": attrs["axis"],
            "use_mean": attrs["use_mean"],
        })
        stats = Node("<layer_norm_stats>", "nn.norm_stats", (), stats_type)
        return NormApply.infer_type(
            (cls.value.read(inputs), stats, cls.scale.read(inputs), cls.bias.read(inputs)),
            attrs,
        )

    @classmethod
    def evaluate(cls, node, arguments, context):
        del context
        value = cls.value.read(arguments)
        stats = norm_stats_value(
            value,
            axis=int(node.attrs["axis"]),
            use_mean=bool(node.attrs["use_mean"]),
        )
        return norm_apply_value(
            value,
            stats,
            cls.scale.read(arguments),
            cls.bias.read(arguments),
            axis=int(node.attrs["axis"]),
            epsilon=float(node.attrs["epsilon"]),
            use_mean=bool(node.attrs["use_mean"]),
            round_before_scale=bool(node.attrs.get("round_before_scale", False)),
        )

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        del node
        return OpCost(notes=("decompose-to-norm-stats-apply",))


__all__ = ["LayerNorm"]
