# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Identity-bearing physical storage objects for bufferized TIR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from triton.flagmega.ir.dim_expr import Dimension, DimensionLike, dim
from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class PhysicalBuffer:
    """One allocation identity, independent from logical Buffer views.

    ``start`` is the allocation's byte start in its memory-space arena and
    ``size`` is its allocation extent.  Distinct PhysicalBuffers may reuse the
    same arena range when their lifetimes do not overlap; that is allocator
    reuse, not semantic aliasing.
    """

    id: str
    memory_space: str
    size: Dimension
    alignment: int
    start: Dimension = dim(0)
    function: str | None = None
    live_start: int | None = None
    live_end: int | None = None
    role: str = "workspace"

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", dim(self.size).simplify())
        object.__setattr__(self, "start", dim(self.start).simplify())
        if not self.id or not self.memory_space:
            raise IRSchemaError("PhysicalBuffer requires non-empty id and memory_space.")
        if isinstance(self.alignment, bool) or self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise IRSchemaError("PhysicalBuffer alignment must be a positive power of two.")
        if self.size.minimum is not None and self.size.minimum < 0:
            raise IRSchemaError("PhysicalBuffer size cannot be negative.")
        if self.start.minimum is not None and self.start.minimum < 0:
            raise IRSchemaError("PhysicalBuffer start cannot be negative.")
        if (self.live_start is None) != (self.live_end is None):
            raise IRSchemaError("PhysicalBuffer lifetime requires both live_start and live_end.")
        if self.live_start is not None and self.live_end < self.live_start:
            raise IRSchemaError("PhysicalBuffer live_end cannot precede live_start.")

    @property
    def nbytes(self) -> int:
        return self.size.fixed_value

    @property
    def offset(self) -> int:
        return self.start.fixed_value

    def with_start(self, start: DimensionLike) -> PhysicalBuffer:
        return replace(self, start=dim(start))

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "memory_space": self.memory_space,
            "size": self.size.to_data(),
            "alignment": self.alignment,
            "start": self.start.to_data(),
            "function": self.function,
            "live_start": self.live_start,
            "live_end": self.live_end,
            "role": self.role,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> PhysicalBuffer:
        return cls(
            id=str(data["id"]),
            memory_space=str(data["memory_space"]),
            size=_dimension_from_data(data.get("size", data.get("nbytes", 0))),
            alignment=int(data["alignment"]),
            start=_dimension_from_data(data.get("start", data.get("offset", 0))),
            function=None if data.get("function") is None else str(data["function"]),
            live_start=None if data.get("live_start") is None else int(data["live_start"]),
            live_end=None if data.get("live_end") is None else int(data["live_end"]),
            role=str(data.get("role", "workspace")),
        )


def _dimension_from_data(value: Any) -> Dimension:
    return Dimension.from_data(value) if isinstance(value, Mapping) else dim(int(value))


# Compatibility spelling for code written against buffer-plan/v2.  Allocation
# is a state of a PhysicalBuffer, not a second kind of storage object.
PhysicalAllocation = PhysicalBuffer


__all__ = ["PhysicalAllocation", "PhysicalBuffer"]
