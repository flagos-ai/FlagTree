# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern for scalar constant expressions."""

from __future__ import annotations

from typing import Callable

from triton.flagmega.ir import Node
from triton.flagmega.ir.type_pattern import TypePattern
from triton.flagmega.pattern_match.pattern import Pattern


class ConstPattern(Pattern):
    def __init__(
        self,
        value: bool | int | float | None = None,
        *,
        condition: Callable[[object], bool] | None = None,
        name: str | None = None,
        type_pattern: TypePattern | None = None,
    ) -> None:
        super().__init__(name, type_pattern=type_pattern)
        if value is not None and condition is not None:
            raise TypeError("ConstPattern accepts either an exact value or condition.")
        self.value = value
        self.condition = condition

    def match_leaf(self, node: Node) -> bool:
        if not super().match_leaf(node) or node.op not in {
            "builtin.scalar_const",
            "tir.scalar_const",
        }:
            return False
        value = node.attrs.get("value")
        return (
            bool(self.condition(value)) if self.condition is not None
            else self.value is None or value == self.value
        )


def is_const(
    value: bool | int | float | None = None,
    *,
    condition: Callable[[object], bool] | None = None,
    name: str | None = None,
    type_pattern: TypePattern | None = None,
) -> ConstPattern:
    return ConstPattern(
        value,
        condition=condition,
        name=name,
        type_pattern=type_pattern,
    )


__all__ = ["ConstPattern", "is_const"]
