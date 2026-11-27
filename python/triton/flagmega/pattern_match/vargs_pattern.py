# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pattern for a fixed or dynamically generated operand list."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from triton.flagmega.ir.model import Node
from triton.flagmega.pattern_match.pattern import Pattern


PatternGenerator = Callable[[Sequence[Node]], Sequence[Pattern]]


class VArgsPattern(Pattern):
    def __init__(
        self,
        fields: Sequence[Pattern] | PatternGenerator,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._generator = fields if callable(fields) else None
        self._fields = None if callable(fields) else tuple(fields)

    def patterns_for(self, nodes: Sequence[Node]) -> tuple[Pattern, ...] | None:
        fields = tuple(self._generator(nodes)) if self._generator is not None else self._fields
        if fields is None or len(fields) != len(nodes):
            return None
        if any(not isinstance(pattern, Pattern) for pattern in fields):
            raise TypeError("VArgsPattern generators must return Pattern objects.")
        return fields

    @property
    def fields(self) -> tuple[Pattern, ...]:
        if self._fields is None:
            raise TypeError("A generated VArgsPattern has no fields before matching.")
        return self._fields

    def match_leaf(self, node: Node) -> bool:
        return False


def is_vargs(*patterns: Pattern, name: str | None = None) -> VArgsPattern:
    return VArgsPattern(patterns, name)


def is_vargs_repeat(pattern_factory: Callable[[], Pattern], name: str | None = None) -> VArgsPattern:
    return VArgsPattern(lambda nodes: tuple(pattern_factory() for _ in nodes), name)


__all__ = ["PatternGenerator", "VArgsPattern", "is_vargs", "is_vargs_repeat"]
