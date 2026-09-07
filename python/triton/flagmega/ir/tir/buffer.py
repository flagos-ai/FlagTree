# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.bufferization import MemSpan
from triton.flagmega.ir.dim_expr import Dimension, dim
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import (
    is_local_shard_subview,
    local_shape,
    placement_owner_count,
)
from triton.flagmega.ir.model import DistributedType, IRType, TensorType
from triton.flagmega.ir.tir.base import TIRNode, tir_node
from triton.flagmega.ir.types import DataType, data_type


@tir_node("buffer")
@dataclass(frozen=True)
class Buffer(TIRNode):
    name: str
    elem_type: DataType
    mem_span: MemSpan
    dimensions: tuple[Dimension, ...]
    strides: tuple[Dimension, ...]
    distributed_type: DistributedType | None = None
    distributed_storage_kind: DistributedBufferStorageKind = DistributedBufferStorageKind.COMPACT_LOCAL
    distributed_backing_type: DistributedType | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "elem_type", data_type(self.elem_type))
        object.__setattr__(self, "dimensions", tuple(dim(value).simplify() for value in self.dimensions))
        object.__setattr__(self, "strides", tuple(dim(value).simplify() for value in self.strides))
        object.__setattr__(
            self,
            "distributed_storage_kind",
            DistributedBufferStorageKind(self.distributed_storage_kind),
        )
        if not self.name:
            raise IRSchemaError("TIR Buffer requires a non-empty name.")
        if len(self.dimensions) != len(self.strides):
            raise IRSchemaError("TIR Buffer dimensions and element strides must have equal rank.")
        if any(value.minimum is not None and value.minimum < 0 for value in self.dimensions):
            raise IRSchemaError("TIR Buffer dimensions cannot be negative.")
        if any(value.minimum is not None and value.minimum < 0 for value in self.strides):
            raise IRSchemaError("TIR Buffer strides cannot be negative.")
        if self.distributed_type is not None:
            logical = self.distributed_type.tensor
            if logical.dtype != self.elem_type or logical.shape != self.dimensions:
                raise IRSchemaError("TIR Buffer DistributedType must match its dtype and dimensions.")
        elif (
            self.distributed_storage_kind is not DistributedBufferStorageKind.COMPACT_LOCAL
            or self.distributed_backing_type is not None
        ):
            raise IRSchemaError("Non-local distributed buffer storage requires a DistributedType.")
        if (
            self.distributed_storage_kind is DistributedBufferStorageKind.CANONICAL_GLOBAL
            and self.mem_span.buffer.memory_space in {"shared", "block_local_data", "block_local_rdata"}
        ):
            raise IRSchemaError("Canonical-global buffers cannot use block-local/shared storage.")
        if (
            self.distributed_storage_kind
            is DistributedBufferStorageKind.REPLICATED_LOCAL
            and (
                self.distributed_type is None
                or self.distributed_type.partial is not None
                or any(
                    level != "b"
                    for level in self.distributed_type.placement.hierarchy_levels
                )
            )
        ):
            raise IRSchemaError(
                "Replicated-local buffers require a non-partial block placement."
            )
        if self.distributed_backing_type is not None:
            backing = self.distributed_backing_type
            if self.distributed_storage_kind not in {
                DistributedBufferStorageKind.COMPACT_LOCAL,
                DistributedBufferStorageKind.COMPACT_PER_OWNER,
            }:
                raise IRSchemaError(
                    "A parent-shard backing is valid only for compact distributed storage."
                )
            if (
                self.distributed_type is None
                or backing.tensor != self.distributed_type.tensor
                or backing.placement != self.distributed_type.placement
                or not is_local_shard_subview(backing, self.distributed_type)
            ):
                raise IRSchemaError(
                    "A parent-shard backing must contain every target local shard."
                )
        if (
            self.distributed_type is not None
            and self.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_OWNER
            and self.mem_span.buffer.size.minimum is not None
            and self.mem_span.size.maximum is not None
            and self.mem_span.buffer.size.minimum
            < self.mem_span.size.maximum * placement_owner_count(self.distributed_type)
        ):
            raise IRSchemaError(
                "Compact-per-owner Buffer backing must contain one component per placement owner."
            )
        if (
            all(value.is_fixed for value in self.dimensions)
            and all(value.is_fixed for value in self.strides)
            and self.mem_span.size.is_fixed
        ):
            storage_dimensions = (
                local_shape(
                    self.distributed_backing_type or self.distributed_type
                )
                if self.distributed_type is not None
                and not self.distributed_storage_kind.exposes_logical_coordinates
                else self.dimensions
            )
            if any(value.fixed_value == 0 for value in storage_dimensions):
                required = 0
            else:
                last_element = sum(
                    (dimension.fixed_value - 1) * stride.fixed_value
                    for dimension, stride in zip(storage_dimensions, self.strides)
                )
                required = (last_element + 1) * self.elem_type.itemsize
            if required > self.mem_span.size.fixed_value:
                raise IRSchemaError(
                    f"TIR Buffer {self.name!r} needs {required} bytes but its MemSpan has "
                    f"{self.mem_span.size.fixed_value}."
                )

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @property
    def type(self) -> IRType:
        return self.distributed_type or TensorType(self.elem_type, self.dimensions)


__all__ = ["Buffer", "DistributedBufferStorageKind"]
