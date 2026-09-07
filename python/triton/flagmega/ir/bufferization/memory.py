# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent memory-space contracts for physical bufferization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from triton.flagmega.errors import IRSchemaError


class AllocationPolicy(str, Enum):
    GRANULARITY_ALIGNED = "granularity_aligned"
    POWER_OF_TWO = "power_of_two"


class AllocationStrategy(str, Enum):
    SAT = "sat"
    LINEAR = "linear"
    EXTERNAL = "external"


class MemoryAllocationScope(str, Enum):
    """Lifetime/ownership of one compiler-managed arena."""

    MODULE = "module"
    FUNCTION = "function"
    EXTERNAL = "external"


class MemorySharingScope(str, Enum):
    """Physical agents which may address the same memory instance.

    This is the Python IR equivalent of nncase ``MemorySharingScope``.  It is
    deliberately independent from ``MemoryAllocationScope``: a function-owned
    arena can either be replicated per block or shared by the complete chip.
    """

    BLOCK = "block"
    DIE = "die"
    CHIP = "chip"


@dataclass(frozen=True)
class MemorySpace:
    """One physical allocation domain supplied by a target machine."""

    name: str
    kind: str
    granularity: int
    maximum_bytes: int
    strategy: AllocationStrategy
    allocation_policy: AllocationPolicy = AllocationPolicy.GRANULARITY_ALIGNED
    allocation_scope: MemoryAllocationScope = MemoryAllocationScope.FUNCTION
    sharing_scope: MemorySharingScope = MemorySharingScope.CHIP

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise IRSchemaError("A memory space requires non-empty name and kind.")
        if self.granularity <= 0 or self.granularity & (self.granularity - 1):
            raise IRSchemaError("Memory-space granularity must be a positive power of two.")
        if self.maximum_bytes <= 0:
            raise IRSchemaError("Memory-space maximum_bytes must be positive.")
        object.__setattr__(
            self, "allocation_scope", MemoryAllocationScope(self.allocation_scope)
        )
        object.__setattr__(self, "sharing_scope", MemorySharingScope(self.sharing_scope))
        if (
            self.strategy is AllocationStrategy.EXTERNAL
            and self.allocation_scope is not MemoryAllocationScope.EXTERNAL
        ):
            raise IRSchemaError(
                "An external memory space requires external allocation ownership."
            )
        if (
            self.allocation_scope is MemoryAllocationScope.EXTERNAL
            and self.strategy is not AllocationStrategy.EXTERNAL
        ):
            raise IRSchemaError(
                "External allocation ownership requires the external strategy."
            )

    @property
    def shared_scope(self) -> str:
        """Compatibility spelling used by buffer-plan/v5 checkpoints."""

        return self.allocation_scope.value

    def allocation_bytes(self, requested_bytes: int) -> int:
        if requested_bytes < 0:
            raise IRSchemaError("Requested allocation size cannot be negative.")
        if requested_bytes == 0:
            return 0
        if self.allocation_policy is AllocationPolicy.GRANULARITY_ALIGNED:
            result = _align_up(requested_bytes, self.granularity)
        elif self.allocation_policy is AllocationPolicy.POWER_OF_TWO:
            result = max(self.granularity, 1 << (requested_bytes - 1).bit_length())
        else:  # pragma: no cover - exhaustive enum guard
            raise IRSchemaError(f"Unsupported allocation policy {self.allocation_policy!r}.")
        if result > self.maximum_bytes:
            raise IRSchemaError(
                f"Memory space {self.name!r} requires {result} bytes after "
                f"{self.allocation_policy.value} rounding, exceeding {self.maximum_bytes}."
            )
        return result

    def to_data(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "granularity": self.granularity,
            "maximum_bytes": self.maximum_bytes,
            "strategy": self.strategy.value,
            "allocation_policy": self.allocation_policy.value,
            "allocation_scope": self.allocation_scope.value,
            "sharing_scope": self.sharing_scope.value,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> MemorySpace:
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            granularity=int(data["granularity"]),
            maximum_bytes=int(data["maximum_bytes"]),
            strategy=AllocationStrategy(str(data["strategy"])),
            allocation_policy=AllocationPolicy(str(data["allocation_policy"])),
            allocation_scope=MemoryAllocationScope(
                str(data.get("allocation_scope", data.get("shared_scope", "function")))
            ),
            sharing_scope=MemorySharingScope(
                str(data.get("sharing_scope", _legacy_sharing_scope(data)))
            ),
        )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _legacy_sharing_scope(data: Mapping[str, Any]) -> str:
    # Old checkpoints did not distinguish arena ownership from physical
    # visibility.  CUDA Shared is block-private; every other currently emitted
    # device/external arena was allocated once for the complete launch.
    return "block" if str(data.get("kind", "")) == "shared" else "chip"


__all__ = [
    "AllocationPolicy",
    "AllocationStrategy",
    "MemoryAllocationScope",
    "MemorySharingScope",
    "MemorySpace",
]
