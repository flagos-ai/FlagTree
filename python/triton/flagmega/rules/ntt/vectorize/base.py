# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Contracts shared by nncase-shaped AutoVectorize rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from triton.flagmega.ir import IRModule, Node
from triton.flagmega.rules import RewriteResult, RewriteRule


@dataclass(frozen=True)
class VectorizeCandidate:
    id: str
    rule: str
    axes: tuple[int, ...]
    lanes: tuple[int, ...]
    parameters: dict[str, object]
    facts: dict[str, object]


class VectorizeRule(Protocol):
    name: str
    op_names: frozenset[str]

    def candidates(self, node: Node, module: IRModule) -> tuple[VectorizeCandidate, ...]: ...

    def rewrite(self, node: Node, module: IRModule, candidate: VectorizeCandidate) -> RewriteResult: ...


class VectorizeRuleRegistry:
    """Explicit target-owned rule registry, equivalent to ``IRulesAddable``."""

    def __init__(self) -> None:
        self._rules: list[VectorizeRule] = []
        self._propagation: list[RewriteRule] = []

    def add(self, rule: VectorizeRule) -> None:
        if any(value.name == rule.name for value in self._rules):
            raise ValueError(f"Vectorize rule {rule.name!r} is already registered.")
        self._rules.append(rule)

    def add_propagation(self, rule: RewriteRule) -> None:
        if any(value.name == rule.name for value in self._propagation):
            raise ValueError(f"Vectorize propagation rule {rule.name!r} is already registered.")
        self._propagation.append(rule)

    @property
    def rules(self) -> tuple[VectorizeRule, ...]:
        return tuple(self._rules)

    @property
    def propagation_rules(self) -> tuple[RewriteRule, ...]:
        return tuple(self._propagation)


__all__ = ["VectorizeCandidate", "VectorizeRule", "VectorizeRuleRegistry"]
