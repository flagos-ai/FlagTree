# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""One caller-owned function memory-pool frame."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("memory_pool_frame")
@dataclass(frozen=True)
class MemoryPoolFrame(TIRNode):
    """Physical sub-allocation passed across one nested function call."""

    memory_space: str
    allocation: str | None
    offset: int
    nbytes: int

    def __post_init__(self) -> None:
        if not self.memory_space:
            raise IRSchemaError("MemoryPoolFrame requires a memory-space identity.")
        if (
            isinstance(self.offset, bool)
            or isinstance(self.nbytes, bool)
            or self.offset < 0
            or self.nbytes < 0
            or (self.nbytes == 0 and self.offset != 0)
        ):
            raise IRSchemaError(
                "MemoryPoolFrame requires non-negative bounds and zero offset "
                "when empty."
            )
        if self.allocation is not None and not self.allocation:
            raise IRSchemaError(
                "MemoryPoolFrame allocation must be non-empty when present."
            )


__all__ = ["MemoryPoolFrame"]
