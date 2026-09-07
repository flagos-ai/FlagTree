# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern for function parameters."""

from __future__ import annotations

from triton.flagmega.ir.model import Node
from triton.flagmega.ir.type_pattern import TypePattern
from triton.flagmega.pattern_match.pattern import Pattern


class VarPattern(Pattern):
    def __init__(self, name: str | None = None, *, type_pattern: TypePattern | None = None) -> None:
        super().__init__(name, type_pattern=type_pattern)

    def match_leaf(self, node: Node) -> bool:
        return super().match_leaf(node) and node.op == "builtin.var"


def is_var(name: str | None = None, *, type_pattern: TypePattern | None = None) -> VarPattern:
    return VarPattern(name, type_pattern=type_pattern)


__all__ = ["VarPattern", "is_var"]
