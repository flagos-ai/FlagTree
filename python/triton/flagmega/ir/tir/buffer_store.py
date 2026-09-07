# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import Dimension, dim
from triton.flagmega.ir.tir.base import TIRCost, TIRStmt, TIRValue, tir_node
from triton.flagmega.ir.tir.buffer import Buffer


@tir_node("buffer_store")
@dataclass(frozen=True)
class BufferStore(TIRStmt):
    buffer: Buffer
    indices: tuple[Dimension, ...]
    value: TIRValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(dim(value).simplify() for value in self.indices))
        if len(self.indices) != self.buffer.rank:
            raise IRSchemaError("TIR BufferStore index rank must equal its Buffer rank.")
        if self.value.type.dtype != self.buffer.elem_type or self.value.type.rank != 0:
            raise IRSchemaError("TIR BufferStore value must be one scalar buffer element.")

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(bytes_written=self.buffer.elem_type.itemsize)


__all__ = ["BufferStore"]
