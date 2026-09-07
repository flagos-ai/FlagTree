# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Aggregate logical value backed by multiple first-class TIR buffers."""

from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import DistributedType, IRType, RefType, TensorType, TupleType, logical_type
from triton.flagmega.ir.tir.base import TIRValue, tir_node
from triton.flagmega.ir.tir.buffer import Buffer


@tir_node("buffer_tuple")
@dataclass(frozen=True)
class BufferTuple(TIRValue):
    buffers: tuple[Buffer, ...]
    value_type: IRType

    def __post_init__(self) -> None:
        object.__setattr__(self, "buffers", tuple(self.buffers))
        if not self.buffers or not isinstance(self.value_type, IRType):
            raise IRSchemaError("BufferTuple requires buffers and an aggregate IRType.")
        leaves = _leaf_types(self.value_type)
        if len(leaves) != len(self.buffers) or any(
            logical_type(buffer.type) != logical_type(leaf)
            for buffer, leaf in zip(self.buffers, leaves)
        ):
            raise IRSchemaError("BufferTuple buffers do not match its logical IRType leaves.")

    @property
    def type(self) -> IRType:
        return self.value_type


__all__ = ["BufferTuple"]


def _leaf_types(value_type):
    if isinstance(value_type, (TensorType, DistributedType)):
        return (value_type,)
    if isinstance(value_type, RefType):
        return tuple(leaf for _, field in value_type.fields for leaf in _leaf_types(field))
    if isinstance(value_type, TupleType):
        return tuple(leaf for field in value_type.fields for leaf in _leaf_types(field))
    raise IRSchemaError(f"BufferTuple cannot flatten {type(value_type).__name__}.")
