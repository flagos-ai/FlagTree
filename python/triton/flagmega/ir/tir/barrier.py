# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit block/chip synchronization in an execution function."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRCost, TIRStmt, tir_node
from triton.flagmega.ir.tir.synchronization_range import SynchronizationRange


class BarrierScope(str, Enum):
    BLOCK = "block"
    CHIP = "chip"


@tir_node("barrier")
@dataclass(frozen=True)
class Barrier(TIRStmt):
    scope: BarrierScope
    after: tuple[str, ...]
    before: str
    hazards: tuple[str, ...] = ()
    ranges: tuple[SynchronizationRange, ...] = ()
    axis_group_axes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", BarrierScope(self.scope))
        object.__setattr__(self, "after", tuple(self.after))
        object.__setattr__(self, "hazards", tuple(self.hazards))
        object.__setattr__(self, "ranges", tuple(self.ranges))
        axes = tuple(sorted(set(int(value) for value in self.axis_group_axes)))
        object.__setattr__(self, "axis_group_axes", axes)
        if not self.after or any(not value for value in self.after) or not self.before:
            raise IRSchemaError("Barrier requires explicit after/before call identities.")
        if self.scope is BarrierScope.BLOCK and self.axis_group_axes:
            raise IRSchemaError("A block barrier cannot carry chip axis groups.")
        if any(value < 0 for value in axes):
            raise IRSchemaError("Barrier axis-group indices must be non-negative.")
        if any(not isinstance(value, SynchronizationRange) for value in self.ranges):
            raise IRSchemaError("Barrier ranges must be SynchronizationRange values.")

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(synchronizations=1)


__all__ = ["Barrier", "BarrierScope"]
