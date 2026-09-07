# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Per-function memory pools and concrete nested-call frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class FunctionMemoryPool:
    memory_space: str
    scope_bytes: int
    alignment: int
    allocations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.memory_space:
            raise IRSchemaError("A function memory pool requires a memory-space identity.")
        if isinstance(self.scope_bytes, bool) or self.scope_bytes < 0:
            raise IRSchemaError("Function memory-pool size cannot be negative.")
        if (
            isinstance(self.alignment, bool)
            or self.alignment <= 0
            or self.alignment & (self.alignment - 1)
        ):
            raise IRSchemaError("Function memory-pool alignment must be a power of two.")
        if len(set(self.allocations)) != len(self.allocations):
            raise IRSchemaError("Function memory-pool allocations must be unique.")

    def to_data(self) -> dict[str, object]:
        return {
            "memory_space": self.memory_space,
            "scope_bytes": self.scope_bytes,
            "alignment": self.alignment,
            "allocations": list(self.allocations),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> FunctionMemoryPool:
        return cls(
            str(data["memory_space"]),
            int(data["scope_bytes"]),
            int(data["alignment"]),
            tuple(str(value) for value in data.get("allocations", ())),
        )


@dataclass(frozen=True)
class CallMemoryPoolBinding:
    memory_space: str
    allocation: str | None
    offset: int
    scope_bytes: int

    def __post_init__(self) -> None:
        if not self.memory_space:
            raise IRSchemaError("A call memory-pool binding requires a memory space.")
        if (
            isinstance(self.offset, bool)
            or isinstance(self.scope_bytes, bool)
            or self.offset < 0
            or self.scope_bytes < 0
        ):
            raise IRSchemaError("Call memory-pool offsets and sizes cannot be negative.")
        if self.allocation is None and (self.offset or self.scope_bytes):
            raise IRSchemaError("A non-empty call memory-pool frame requires an allocation.")

    def to_data(self) -> dict[str, object]:
        return {
            "memory_space": self.memory_space,
            "allocation": self.allocation,
            "offset": self.offset,
            "scope_bytes": self.scope_bytes,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> CallMemoryPoolBinding:
        return cls(
            str(data["memory_space"]),
            None if data.get("allocation") is None else str(data["allocation"]),
            int(data.get("offset", 0)),
            int(data.get("scope_bytes", 0)),
        )


__all__ = ["CallMemoryPoolBinding", "FunctionMemoryPool"]
