# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical byte range protected by a TIR barrier."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.tir.base import TIRNode, tir_node


@tir_node("synchronization_range")
@dataclass(frozen=True)
class SynchronizationRange(TIRNode):
    storage: str
    physical_id: str
    offset: int
    nbytes: int
    mode: str

    def __post_init__(self) -> None:
        if not self.storage or not self.physical_id:
            raise IRSchemaError("SynchronizationRange requires storage identity.")
        if self.offset < 0 or self.nbytes <= 0:
            raise IRSchemaError(
                "SynchronizationRange requires a non-negative offset and positive size."
            )
        if self.mode not in {"read", "write", "read_write"}:
            raise IRSchemaError(
                "SynchronizationRange mode must be read, write, or read_write."
            )


__all__ = ["SynchronizationRange"]
