# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""First-class byte spans over PhysicalBuffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from triton.flagmega.ir.bufferization.physical_buffer import PhysicalBuffer
from triton.flagmega.ir.dim_expr import Dimension, DimensionLike, dim, equivalent_dim
from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class MemSpan:
    """A half-open relative byte interval in one PhysicalBuffer."""

    buffer: PhysicalBuffer
    start: Dimension = dim(0)
    size: Dimension | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", dim(self.start).simplify())
        object.__setattr__(
            self,
            "size",
            self.buffer.size if self.size is None else dim(self.size).simplify(),
        )
        if self.start.minimum is not None and self.start.minimum < 0:
            raise IRSchemaError("MemSpan start cannot be negative.")
        if self.size.minimum is not None and self.size.minimum < 0:
            raise IRSchemaError("MemSpan size cannot be negative.")
        overflow = (self.end - self.buffer.size).simplify()
        if overflow.minimum is not None and overflow.minimum > 0:
            raise IRSchemaError(
                f"MemSpan [{self.start}, {self.end}) exceeds PhysicalBuffer "
                f"{self.buffer.id!r} size {self.buffer.size}."
            )

    @property
    def end(self) -> Dimension:
        return (self.start + self.size).simplify()

    @property
    def absolute_start(self) -> Dimension:
        return (self.buffer.start + self.start).simplify()

    @property
    def absolute_end(self) -> Dimension:
        return (self.absolute_start + self.size).simplify()

    @property
    def byte_offset(self) -> int:
        return self.start.fixed_value

    @property
    def nbytes(self) -> int:
        return self.size.fixed_value

    @property
    def offset(self) -> int:
        return self.absolute_start.fixed_value

    def subspan(
        self,
        start: DimensionLike,
        size: DimensionLike | None = None,
    ) -> MemSpan:
        relative = dim(start)
        return MemSpan(
            self.buffer,
            (self.start + relative).simplify(),
            (self.size - relative).simplify() if size is None else dim(size),
        )

    def must_alias(self, other: MemSpan) -> bool:
        return (
            self.buffer.id == other.buffer.id
            and equivalent_dim(self.start, other.start)
            and equivalent_dim(self.size, other.size)
        )

    def may_alias(self, other: MemSpan) -> bool:
        if self.buffer.id != other.buffer.id:
            return False
        if _known_nonpositive(self.size) or _known_nonpositive(other.size):
            return False
        # Unknown symbolic ordering is conservatively aliasing.  A false
        # negative would make rewrite/lifetime/synchronization unsound.
        return not (
            _provably_le(self.end, other.start)
            or _provably_le(other.end, self.start)
        )

    def is_within(self, other: MemSpan) -> bool:
        return (
            self.buffer.id == other.buffer.id
            and _provably_le(other.start, self.start)
            and _provably_le(self.end, other.end)
        )

    def to_data(self) -> dict[str, object]:
        return {
            "buffer": self.buffer.id,
            "start": self.start.to_data(),
            "size": self.size.to_data(),
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        physical_buffers: Mapping[str, PhysicalBuffer],
    ) -> MemSpan:
        buffer_id = str(data["buffer"])
        try:
            buffer = physical_buffers[buffer_id]
        except KeyError as error:
            raise ValueError(f"MemSpan references missing PhysicalBuffer {buffer_id!r}.") from error
        return cls(
            buffer,
            _dimension_from_data(data.get("start", 0)),
            _dimension_from_data(data.get("size", buffer.size.to_data())),
        )


def _known_nonpositive(value: Dimension) -> bool:
    return value.maximum is not None and value.maximum <= 0


def _provably_le(lhs: Dimension, rhs: Dimension) -> bool:
    difference = (rhs - lhs).simplify()
    return difference.minimum is not None and difference.minimum >= 0


def _dimension_from_data(value: Any) -> Dimension:
    return Dimension.from_data(value) if isinstance(value, Mapping) else dim(int(value))


__all__ = ["MemSpan"]
