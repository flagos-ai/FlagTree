# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed NVIDIA SM90 machine capabilities used by selection and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class Sm90Capability:
    compute_capability: tuple[int, int] = (9, 0)
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 2048
    max_registers_per_sm: int = 65536
    max_shared_memory_bytes: int = 227328
    supports_fp8_mma: bool = True
    supports_async_copy: bool = True
    supports_cooperative_grid: bool = True
    supports_tma: bool = True
    supports_warp_specialize: bool = True
    supports_grid_sync: bool = True
    supports_mma_v3: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute_capability", tuple(self.compute_capability))
        if (
            len(self.compute_capability) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in self.compute_capability)
            or self.compute_capability[0] != 9
        ):
            raise IRSchemaError(
                "Sm90Capability requires a compute capability in the 9.x family."
            )
        for name in (
            "warp_size",
            "max_threads_per_block",
            "max_threads_per_sm",
            "max_registers_per_sm",
            "max_shared_memory_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise IRSchemaError(f"Sm90Capability {name} must be positive.")
        for name in (
            "supports_fp8_mma",
            "supports_async_copy",
            "supports_cooperative_grid",
            "supports_tma",
            "supports_warp_specialize",
            "supports_grid_sync",
            "supports_mma_v3",
        ):
            if not isinstance(getattr(self, name), bool):
                raise IRSchemaError(f"Sm90Capability {name} must be bool.")
        if self.max_threads_per_block > self.max_threads_per_sm:
            raise IRSchemaError(
                "SM90 max_threads_per_block cannot exceed max_threads_per_sm."
            )

    @property
    def features(self) -> frozenset[str]:
        # ``fp8`` describes scalar conversion/storage support used by portable
        # Triton algorithms; ``fp8_mma`` is the stronger matrix-instruction
        # capability. Architecture identity is serialized separately.
        values = {"fp8"}
        for enabled, name in (
            (self.supports_fp8_mma, "fp8_mma"),
            (self.supports_async_copy, "async_copy"),
            (self.supports_cooperative_grid, "cooperative_grid"),
            (self.supports_tma, "tma"),
            (self.supports_warp_specialize, "warp_specialize"),
            (self.supports_grid_sync, "grid_sync"),
            (self.supports_mma_v3, "mma_v3"),
        ):
            if enabled:
                values.add(name)
        return frozenset(values)

    def missing(self, requirements: Iterable[object]) -> tuple[str, ...]:
        required = {str(value) for value in requirements}
        return tuple(sorted(required - self.features))

    def supports(self, requirements: Iterable[object]) -> bool:
        return not self.missing(requirements)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "flagmega.nvidia-sm90-capability/v1",
            "compute_capability": list(self.compute_capability),
            "warp_size": self.warp_size,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_sm": self.max_threads_per_sm,
            "max_registers_per_sm": self.max_registers_per_sm,
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "supports_fp8_mma": self.supports_fp8_mma,
            "supports_async_copy": self.supports_async_copy,
            "supports_cooperative_grid": self.supports_cooperative_grid,
            "supports_tma": self.supports_tma,
            "supports_warp_specialize": self.supports_warp_specialize,
            "supports_grid_sync": self.supports_grid_sync,
            "supports_mma_v3": self.supports_mma_v3,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> Sm90Capability:
        if data.get("schema") != "flagmega.nvidia-sm90-capability/v1":
            raise IRSchemaError(
                f"Unsupported SM90 capability schema {data.get('schema')!r}."
            )
        return cls(
            compute_capability=tuple(data["compute_capability"]),
            warp_size=data["warp_size"],
            max_threads_per_block=data["max_threads_per_block"],
            max_threads_per_sm=data["max_threads_per_sm"],
            max_registers_per_sm=data["max_registers_per_sm"],
            max_shared_memory_bytes=data["max_shared_memory_bytes"],
            supports_fp8_mma=data["supports_fp8_mma"],
            supports_async_copy=data["supports_async_copy"],
            supports_cooperative_grid=data["supports_cooperative_grid"],
            supports_tma=data["supports_tma"],
            supports_warp_specialize=data["supports_warp_specialize"],
            supports_grid_sync=data["supports_grid_sync"],
            supports_mma_v3=data["supports_mma_v3"],
        )


__all__ = ["Sm90Capability"]
