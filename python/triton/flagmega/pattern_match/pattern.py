# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Base graph-pattern protocol."""

from __future__ import annotations

from typing import Callable

from triton.flagmega.ir.model import Node
from triton.flagmega.ir.type_pattern import TypePattern


class Pattern:
    """Identity-keyed pattern with optional name and result type predicate."""

    def __init__(self, name: str | None = None, *, type_pattern: TypePattern | None = None) -> None:
        self.name = name
        self.type_pattern = type_pattern
        self.user_count: int | None = None

    def match_leaf(self, node: Node) -> bool:
        return self.type_pattern is None or self.type_pattern.match_leaf(node.type)

    def __or__(self, other: Pattern) -> Pattern:
        if not isinstance(other, Pattern):
            return NotImplemented
        from triton.flagmega.pattern_match.or_pattern import OrPattern

        return OrPattern(self, other)

    def with_user_count(self, count: int | None) -> Pattern:
        """Require an exact number of SSA/function-output users."""

        if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
            raise ValueError("Pattern user count must be a non-negative integer or None.")
        self.user_count = count
        return self

    __hash__ = object.__hash__


class ExprPattern(Pattern):
    """Wildcard or predicate pattern over any FlagMega node."""

    def __init__(
        self,
        condition: Callable[[Node], bool] | None = None,
        name: str | None = None,
        *,
        type_pattern: TypePattern | None = None,
    ) -> None:
        super().__init__(name, type_pattern=type_pattern)
        self.condition = condition or (lambda node: True)
        self.is_wildcard = condition is None

    def match_leaf(self, node: Node) -> bool:
        return super().match_leaf(node) and bool(self.condition(node))


def wildcard(
    name: str | None = None,
    condition: Callable[[Node], bool] | None = None,
    *,
    type_pattern: TypePattern | None = None,
) -> ExprPattern:
    return ExprPattern(condition, name, type_pattern=type_pattern)


__all__ = ["ExprPattern", "Pattern", "wildcard"]
