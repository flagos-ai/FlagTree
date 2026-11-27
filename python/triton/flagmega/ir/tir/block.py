# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.types import DType
from triton.flagmega.ir.tir.base import TIRStmt, TIRValue, tir_node
from triton.flagmega.ir.tir.buffer import Buffer
from triton.flagmega.ir.tir.buffer_region import BufferRegion
from triton.flagmega.ir.tir.scalar_var import ScalarVar
from triton.flagmega.ir.tir.sequential import Sequential


@tir_node("block")
@dataclass(frozen=True)
class Block(TIRStmt):
    name: str
    body: Sequential
    init_body: Sequential = Sequential()
    iter_vars: tuple[ScalarVar, ...] = ()
    reads: tuple[BufferRegion, ...] = ()
    writes: tuple[BufferRegion, ...] = ()
    alloc_buffers: tuple[Buffer, ...] = ()
    predicate: TIRValue | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise IRSchemaError("TIR Block requires a non-empty name.")
        for field in ("iter_vars", "reads", "writes", "alloc_buffers"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.predicate is not None and (
            self.predicate.type.rank != 0 or self.predicate.type.dtype != DType.BOOL
        ):
            raise IRSchemaError("TIR Block predicate must be a scalar bool.")


__all__ = ["Block"]
