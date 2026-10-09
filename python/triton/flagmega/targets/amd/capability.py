# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed AMD RDNA machine capabilities used by selection and verification.

FlagMega-on-Radeon T0: capability descriptors for Radeon PRO W7900 (RDNA3 /
gfx1100) and Radeon AI PRO R9700 (RDNA4 / gfx1201). Mirrors the NVIDIA SM90
descriptor surface (``features`` / ``missing`` / ``supports`` / ``to_data`` /
``from_data``) so the backend-agnostic selection policy can consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from triton.flagmega.errors import IRSchemaError


@dataclass(frozen=True)
class AmdCapability:
    # gfx arch id, e.g. "gfx1100" (RDNA3) / "gfx1201" (RDNA4). No fixed-family
    # assertion (unlike SM90's 9.x check) — any gfx11xx/gfx12xx is accepted.
    gfx_arch: str = "gfx1100"
    warp_size: int = 32  # wave32
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 2048
    max_registers_per_sm: int = 65536
    max_shared_memory_bytes: int = 65536  # 64 KB LDS
    supports_wmma: bool = True
    supports_fp8: bool = False
    supports_fp8_mma: bool = False
    supports_async_copy: bool = True
    supports_cooperative_grid: bool = True
    supports_tma: bool = False  # no TMA on RDNA
    supports_warp_specialize: bool = True  # software wave-id partition
    supports_grid_sync: bool = True  # global-memory grid barrier
    supports_mma_v3: bool = False  # no Hopper-style WGMMA

    def __post_init__(self) -> None:
        if not isinstance(self.gfx_arch, str) or not self.gfx_arch:
            raise IRSchemaError("AmdCapability requires a non-empty gfx_arch string.")
        for name in (
            "warp_size",
            "max_threads_per_block",
            "max_threads_per_sm",
            "max_registers_per_sm",
            "max_shared_memory_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise IRSchemaError(f"AmdCapability {name} must be positive.")
        for name in (
            "supports_wmma",
            "supports_fp8",
            "supports_fp8_mma",
            "supports_async_copy",
            "supports_cooperative_grid",
            "supports_tma",
            "supports_warp_specialize",
            "supports_grid_sync",
            "supports_mma_v3",
        ):
            if not isinstance(getattr(self, name), bool):
                raise IRSchemaError(f"AmdCapability {name} must be bool.")
        if self.max_threads_per_block > self.max_threads_per_sm:
            raise IRSchemaError(
                "AMD max_threads_per_block cannot exceed max_threads_per_sm."
            )

    @property
    def features(self) -> frozenset[str]:
        values = set()
        for enabled, name in (
            (self.supports_wmma, "wmma"),
            (self.supports_fp8, "fp8"),
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
            "schema": "flagmega.amd-rdna-capability/v1",
            "gfx_arch": self.gfx_arch,
            "warp_size": self.warp_size,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_sm": self.max_threads_per_sm,
            "max_registers_per_sm": self.max_registers_per_sm,
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "supports_wmma": self.supports_wmma,
            "supports_fp8": self.supports_fp8,
            "supports_fp8_mma": self.supports_fp8_mma,
            "supports_async_copy": self.supports_async_copy,
            "supports_cooperative_grid": self.supports_cooperative_grid,
            "supports_tma": self.supports_tma,
            "supports_warp_specialize": self.supports_warp_specialize,
            "supports_grid_sync": self.supports_grid_sync,
            "supports_mma_v3": self.supports_mma_v3,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> "AmdCapability":
        if data.get("schema") != "flagmega.amd-rdna-capability/v1":
            raise IRSchemaError(
                f"Unsupported AMD capability schema {data.get('schema')!r}."
            )
        fields = {k: v for k, v in data.items() if k != "schema"}
        return cls(**fields)


def Gfx1100Capability() -> AmdCapability:
    """Radeon PRO W7900 — RDNA3, wave32, 64 KB LDS, WMMA (no FP8 matrix)."""
    return AmdCapability(gfx_arch="gfx1100")


def Gfx1201Capability() -> AmdCapability:
    """Radeon AI PRO R9700 — RDNA4, wave32, 64 KB LDS, WMMA + FP8."""
    return AmdCapability(gfx_arch="gfx1201", supports_fp8=True, supports_fp8_mma=True)


__all__ = ["AmdCapability", "Gfx1100Capability", "Gfx1201Capability"]
