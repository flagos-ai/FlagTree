# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class optional operand sentinel, equivalent to nncase ``None``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.ir.model import IRType, Node, NoneType
from triton.flagmega.ir.ops.core import OpCost, OpDefinition, op_definition


@op_definition(
    "builtin.none",
    namespace="builtin",
    functional_name="none",
    display_name="None",
)
class NoneValue(OpDefinition):
    constant_source = True
    const_evaluable = True

    @classmethod
    def infer_type(cls, inputs: Sequence[Node], attrs: Mapping[str, object]) -> IRType:
        return NoneType()

    @classmethod
    def evaluate(cls, node, arguments, context):
        return None

    @classmethod
    def cost(cls, node: Node) -> OpCost:
        return OpCost.exact_zero(notes=("optional-none",))


__all__ = ["NoneValue"]
