# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Named SSA/storage value used by first-class TIR kernel bodies."""

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRType
from triton.flagmega.ir.tir.base import TIRValue, tir_node


@tir_node("value_ref")
@dataclass(frozen=True)
class ValueRef(TIRValue):
    """Reference an ABI value without inventing a physical buffer early.

    Bufferization may later replace this logical reference with a ``Buffer``
    backed by a concrete ``MemSpan``.  Keeping the pre-bufferization form
    typed makes selected TIR independently serializable and verifiable.
    """

    name: str
    value_type: IRType

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.value_type, IRType):
            raise IRSchemaError("TIR ValueRef requires a name and IRType.")

    @property
    def type(self) -> IRType:
        return self.value_type


__all__ = ["ValueRef"]
