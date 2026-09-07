# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered alternative pattern."""

from __future__ import annotations

from triton.flagmega.ir.model import Node
from triton.flagmega.pattern_match.pattern import Pattern


class OrPattern(Pattern):
    def __init__(self, lhs: Pattern, rhs: Pattern, name: str | None = None) -> None:
        super().__init__(name)
        self.lhs = lhs
        self.rhs = rhs

    def match_leaf(self, node: Node) -> bool:
        return self.lhs.match_leaf(node) or self.rhs.match_leaf(node)


def is_alt(*patterns: Pattern, name: str | None = None) -> Pattern:
    if len(patterns) < 2:
        raise ValueError("is_alt requires at least two patterns.")
    result: Pattern = patterns[0]
    for pattern in patterns[1:]:
        result = OrPattern(result, pattern, name if pattern is patterns[-1] else None)
    return result


__all__ = ["OrPattern", "is_alt"]
