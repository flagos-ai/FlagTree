# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable post-bufferization memory synchronization plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SYNCHRONIZATION_SCHEMA = "flagmega.memory-synchronization/v1"


@dataclass(frozen=True)
class MemoryRange:
    storage: str
    physical_id: str
    offset: int
    nbytes: int
    access: str

    def to_data(self) -> dict[str, object]:
        return {
            "storage": self.storage,
            "physical_id": self.physical_id,
            "offset": self.offset,
            "nbytes": self.nbytes,
            "access": self.access,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> MemoryRange:
        return cls(
            str(data["storage"]), str(data["physical_id"]), int(data["offset"]),
            int(data["nbytes"]), str(data["access"]),
        )


@dataclass(frozen=True)
class SynchronizationEvent:
    function: str
    after: str
    before: str
    scope: str
    hazards: tuple[str, ...]
    ranges: tuple[MemoryRange, ...]
    axis_group_axes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        axes = tuple(sorted(set(int(value) for value in self.axis_group_axes)))
        if any(value < 0 for value in axes):
            raise ValueError("Synchronization axis-group axes must be non-negative.")
        if self.scope != "grid" and axes:
            raise ValueError("Only grid synchronization can carry axis-group axes.")
        object.__setattr__(self, "axis_group_axes", axes)

    def to_data(self) -> dict[str, object]:
        return {
            "function": self.function,
            "after": self.after,
            "before": self.before,
            "scope": self.scope,
            "hazards": list(self.hazards),
            "ranges": [value.to_data() for value in self.ranges],
            "axis_group_axes": list(self.axis_group_axes),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> SynchronizationEvent:
        return cls(
            str(data["function"]), str(data["after"]), str(data["before"]),
            str(data["scope"]), tuple(str(value) for value in data.get("hazards", ())),
            tuple(MemoryRange.from_data(value) for value in data.get("ranges", ())),
            tuple(int(value) for value in data.get("axis_group_axes", ())),
        )


@dataclass(frozen=True)
class MemorySynchronizationPlan:
    events: tuple[SynchronizationEvent, ...]

    def to_data(self) -> dict[str, object]:
        return {
            "schema": SYNCHRONIZATION_SCHEMA,
            "events": [value.to_data() for value in self.events],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> MemorySynchronizationPlan:
        return cls(tuple(
            SynchronizationEvent.from_data(value) for value in data.get("events", ())
        ))


__all__ = [
    "MemoryRange",
    "MemorySynchronizationPlan",
    "SYNCHRONIZATION_SCHEMA",
    "SynchronizationEvent",
]
