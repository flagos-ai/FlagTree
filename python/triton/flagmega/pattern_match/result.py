# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Immutable match result indexed by pattern identity or capture name."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from triton.flagmega.pattern_match.pattern import Pattern


class MatchResult(Mapping[Pattern, object]):
    def __init__(self, root: object, matches: Mapping[Pattern, object]) -> None:
        self.root = root
        self._pattern_map = MappingProxyType(dict(matches))
        names: dict[str, object] = {}
        for pattern, value in self._pattern_map.items():
            if pattern.name is None:
                continue
            if pattern.name in names and names[pattern.name] != value:
                raise ValueError(f"Capture name {pattern.name!r} matched multiple values.")
            names[pattern.name] = value
        self._name_map = MappingProxyType(names)

    def __getitem__(self, key: Pattern | str) -> object:
        return self._name_map[key] if isinstance(key, str) else self._pattern_map[key]

    def __iter__(self) -> Iterator[Pattern]:
        return iter(self._pattern_map)

    def __len__(self) -> int:
        return len(self._pattern_map)

    def get_value_or_default(self, name: str, default: object = None) -> object:
        return self._name_map.get(name, default)

    @property
    def captures(self) -> Mapping[str, object]:
        return self._name_map


__all__ = ["MatchResult"]
