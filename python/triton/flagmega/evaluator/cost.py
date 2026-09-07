# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Coverage diagnostics for op-local cost/metric handlers."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.evaluator.support import executable_nodes
from triton.flagmega.ir import CostKind, IRModule, OpCost, get_cost


@dataclass(frozen=True)
class CostCoverageEntry:
    node_id: str
    op: str
    owner: str
    cost: OpCost


@dataclass(frozen=True)
class CostCoverage:
    """Per-node estimates without collapsing unknown factors into zeros."""

    entries: tuple[CostCoverageEntry, ...]

    @property
    def complete(self) -> tuple[CostCoverageEntry, ...]:
        return tuple(value for value in self.entries if value.cost.is_complete)

    @property
    def partial(self) -> tuple[CostCoverageEntry, ...]:
        return tuple(
            value
            for value in self.entries
            if not value.cost.is_complete and value.cost.kind is not CostKind.UNKNOWN
        )

    @property
    def unknown(self) -> tuple[CostCoverageEntry, ...]:
        return tuple(value for value in self.entries if value.cost.kind is CostKind.UNKNOWN)


def inspect_cost_coverage(module: IRModule) -> CostCoverage:
    return CostCoverage(tuple(
        CostCoverageEntry(node.id, node.op, owner, get_cost(node))
        for node, owner in executable_nodes(module)
    ))


__all__ = ["CostCoverage", "CostCoverageEntry", "inspect_cost_coverage"]
