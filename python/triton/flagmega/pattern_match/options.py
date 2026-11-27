# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""State carried across deterministic pattern-match attempts."""

from __future__ import annotations

from triton.flagmega.ir.model import Node
from triton.flagmega.pattern_match.pattern import Pattern


class MatchOptions:
    def __init__(self) -> None:
        # nncase keys this table by expression reference identity.  Node ids are
        # only module-local, so using their spelling here would leak a
        # suppression when one MatchOptions instance is used with two modules.
        self._suppressed: dict[int, set[Pattern]] = {}

    def is_suppressed(self, node: Node, pattern: Pattern) -> bool:
        return pattern in self._suppressed.get(id(node), ())

    def suppress(self, node: Node, pattern: Pattern) -> None:
        self._suppressed.setdefault(id(node), set()).add(pattern)

    def inherit(self, source: Node, destination: Node) -> None:
        if source is destination:
            return
        patterns = self._suppressed.get(id(source))
        if patterns:
            self._suppressed.setdefault(id(destination), set()).update(patterns)


__all__ = ["MatchOptions"]
