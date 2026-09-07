# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class tuple construction for rewrites with multiple logical values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.ir.model import IRType, Node, TupleType
from triton.flagmega.ir.ops.core import (
    OpCost,
    OpDefinition,
    op_definition,
    variadic_input_parameter,
)
from triton.flagmega.ir.type_pattern import is_ir_type


@op_definition(
    "builtin.tuple",
    namespace="builtin",
    functional_name="tuple",
    display_name="Tuple",
)
class TupleValue(OpDefinition):
    const_evaluable = True
    fields = variadic_input_parameter(is_ir_type())

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return TupleType(tuple(value.type for value in inputs))

    @classmethod
    def evaluate(cls, node, arguments, context):
        return tuple(arguments)

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost.exact_zero(notes=("tuple-construction",))


__all__ = ["TupleValue"]
