# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Associate materialized statistics with their source tensor."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, IRType, Node
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    attribute_parameter,
    input_parameter,
    op_definition,
)
from triton.flagmega.ir.ops.nn.norm_stats import NormStats
from triton.flagmega.ir.type_pattern import is_tensor


@op_definition(
    "nn.bind_norm_stats",
    namespace="nn",
    functional_name="bind_norm_stats",
    display_name="NN.BindNormStats",
)
class BindNormStats(OpDefinition):
    value = input_parameter(is_tensor(), name="input")
    stats = input_parameter(is_tensor())
    axis = attribute_parameter()
    use_mean = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attributes: Mapping[str, object]) -> dict[str, object]:
        attrs = super().normalize_attrs(attributes)
        axis = attrs["axis"]
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise IRSchemaError("BindNormStats axis must be an integer.")
        return {"axis": axis, "use_mean": bool(attrs["use_mean"])}

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        value = cls.value.read(inputs)
        stats_type = cls.stats.type_of(inputs)
        expected = NormStats.infer_type((value,), attrs)
        compatible = expected == stats_type
        if (
            not compatible
            and isinstance(expected, DistributedType)
            and expected.partial is not None
            and isinstance(stats_type, DistributedType)
        ):
            compatible = replace(expected, partial=None) == stats_type
        if not compatible:
            raise IRSchemaError(
                f"BindNormStats stats type {stats_type!r} must be materialized NormStats type {expected!r}.")
        if isinstance(stats_type, DistributedType) and stats_type.partial is not None:
            raise IRSchemaError("BindNormStats requires materialized non-partial statistics.")
        return stats_type

    @classmethod
    def evaluate(cls, node, arguments, context):
        del node, context
        return cls.stats.read(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        del node
        return OpCost.exact_zero(notes=("normalization-statistics-binding",))


__all__ = ["BindNormStats"]
