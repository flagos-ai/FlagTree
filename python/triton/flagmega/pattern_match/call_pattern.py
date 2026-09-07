# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern for a FlagMega call node and its operands."""

from __future__ import annotations

from triton.flagmega.ir.model import Node
from triton.flagmega.pattern_match.op_pattern import OpPattern
from triton.flagmega.pattern_match.pattern import Pattern
from triton.flagmega.pattern_match.vargs_pattern import VArgsPattern


class CallPattern(Pattern):
    def __init__(
        self,
        target: OpPattern,
        arguments: VArgsPattern,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.target = target
        self.arguments = arguments

    def match_leaf(self, node: Node) -> bool:
        return super().match_leaf(node) and self.target.match_leaf(node)

    def __getitem__(self, parameter):
        """Address an operand pattern with its declaring ``ParameterInfo``."""

        if parameter.input_index is None:
            raise KeyError(f"Parameter {parameter.name!r} is not an input parameter.")
        return self.arguments.fields[parameter.input_index]


def is_call(
    target: OpPattern,
    *arguments: Pattern,
    name: str | None = None,
) -> CallPattern:
    return CallPattern(target, VArgsPattern(arguments), name)


__all__ = ["CallPattern", "is_call"]
