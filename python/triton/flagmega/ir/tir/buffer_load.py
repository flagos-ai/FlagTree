# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import Dimension, dim
from triton.flagmega.ir.model import TensorType
from triton.flagmega.ir.tir.base import TIRCost, TIRValue, tir_node
from triton.flagmega.ir.tir.buffer import Buffer


@tir_node("buffer_load")
@dataclass(frozen=True)
class BufferLoad(TIRValue):
    buffer: Buffer
    indices: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(dim(value).simplify() for value in self.indices))
        if len(self.indices) != self.buffer.rank:
            raise IRSchemaError("TIR BufferLoad index rank must equal its Buffer rank.")

    @property
    def type(self) -> TensorType:
        return TensorType(self.buffer.elem_type, ())

    @property
    def local_cost(self) -> TIRCost:
        return TIRCost(bytes_read=self.buffer.elem_type.itemsize)


__all__ = ["BufferLoad"]
