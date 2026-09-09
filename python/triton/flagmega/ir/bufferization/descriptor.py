# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Logical buffers as typed views over physical storage."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping

from triton.flagmega.ir.bufferization.alias import AliasInfo, AliasKind
from triton.flagmega.ir.bufferization.mem_span import MemSpan
from triton.flagmega.ir.bufferization.physical_buffer import PhysicalAllocation, PhysicalBuffer
from triton.flagmega.ir.dim_expr import DimensionLike
from triton.flagmega.ir.distributed_storage import DistributedBufferStorageKind
from triton.flagmega.ir.distributed_type import (
    is_local_shard_subview,
    local_shape,
    placement_owner_count,
)
from triton.flagmega.ir.model import DistributedType, type_from_data
from triton.flagmega.ir.types import DataType, data_type_from_data, data_type_to_data


@dataclass(frozen=True)
class BufferDescriptor:
    id: str
    dtype: DataType
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    storage: str
    alignment: int
    mem_span: MemSpan
    source_node: str | None = None
    field: str | None = None
    alias: AliasInfo | None = None
    weight_key: str | None = None
    rdata_group: str | None = None
    group_index: int | None = None
    group_count: int | None = None
    live_start: int | None = None
    live_end: int | None = None
    function: str | None = None
    role: str = "value"
    distributed_type: DistributedType | None = None
    distributed_storage_kind: DistributedBufferStorageKind = (
        DistributedBufferStorageKind.COMPACT_LOCAL
    )
    distributed_backing_type: DistributedType | None = None
    # Bind symbols in MemSpan.start to scalar buffer identities. The MemSpan
    # remains the only address/alias truth; this supplies executable SSA uses.
    offset_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))
        object.__setattr__(self, "strides", tuple(int(value) for value in self.strides))
        object.__setattr__(self, "offset_bindings",
                           tuple((str(symbol), str(value)) for symbol, value in self.offset_bindings))
        if len(dict(self.offset_bindings)) != len(self.offset_bindings) or any(
                not symbol.isidentifier() or not value for symbol, value in self.offset_bindings):
            raise ValueError("Buffer offset bindings require unique symbols and scalar buffer identities.")
        object.__setattr__(
            self,
            "distributed_storage_kind",
            DistributedBufferStorageKind(self.distributed_storage_kind),
        )
        if len(self.shape) != len(self.strides):
            raise ValueError("BufferDescriptor shape/stride ranks must match.")
        if self.distributed_type is None:
            if (
                self.distributed_storage_kind
                is not DistributedBufferStorageKind.COMPACT_LOCAL
                or self.distributed_backing_type is not None
            ):
                raise ValueError("Distributed buffer storage requires a DistributedType.")
            return
        logical = self.distributed_type.tensor
        logical_shape = tuple(
            value.fixed_value if value.is_fixed else value.maximum
            for value in logical.shape
        )
        if logical.dtype != self.dtype or None in logical_shape or tuple(logical_shape) != self.shape:
            raise ValueError("BufferDescriptor DistributedType must match its logical dtype/shape.")
        if (
            self.distributed_type.partial is not None
            and self.distributed_storage_kind is DistributedBufferStorageKind.CANONICAL_GLOBAL
        ):
            raise ValueError(
                "Partial buffer storage requires independent owner components, "
                "not one canonical-global tensor."
            )
        if (
            self.distributed_storage_kind
            is DistributedBufferStorageKind.REPLICATED_LOCAL
            and (
                self.distributed_type.partial is not None
                or any(
                    level != "b"
                    for level in self.distributed_type.placement.hierarchy_levels
                )
            )
        ):
            raise ValueError(
                "Replicated-local storage requires a non-partial block placement."
            )
        if self.distributed_backing_type is not None:
            backing = self.distributed_backing_type
            if self.distributed_storage_kind not in {
                DistributedBufferStorageKind.COMPACT_LOCAL,
                DistributedBufferStorageKind.COMPACT_PER_OWNER,
            }:
                raise ValueError(
                    "A parent-shard backing is valid only for compact distributed storage."
                )
            if (
                backing.tensor != self.distributed_type.tensor
                or backing.placement != self.distributed_type.placement
                or not is_local_shard_subview(backing, self.distributed_type)
            ):
                raise ValueError(
                    "A parent-shard backing must contain every target local shard."
                )
        component_shape = self.component_shape
        component_bytes = prod(component_shape, start=1) * self.dtype.itemsize
        global_bytes = prod(self.shape, start=1) * self.dtype.itemsize
        expected_span = (
            global_bytes
            if self.distributed_storage_kind.exposes_logical_coordinates
            else component_bytes
        )
        if self.mem_span.nbytes != expected_span:
            raise ValueError(
                f"BufferDescriptor {self.id!r} {self.distributed_storage_kind.value} span "
                f"must contain {expected_span} bytes, got {self.mem_span.nbytes}."
            )
        if (
            self.distributed_storage_kind is DistributedBufferStorageKind.COMPACT_PER_OWNER
            and self.mem_span.buffer.nbytes
            < self.mem_span.byte_offset + component_bytes * placement_owner_count(self.distributed_type)
        ):
            raise ValueError(
                f"Compact-per-owner buffer {self.id!r} backing does not contain every owner component."
            )

    @property
    def local_shape(self) -> tuple[int, ...]:
        if self.distributed_type is None:
            return self.shape
        result = []
        for value in local_shape(self.distributed_type):
            extent = value.fixed_value if value.is_fixed else value.maximum
            if extent is None:
                raise ValueError(f"BufferDescriptor {self.id!r} has an unbounded local shape.")
            result.append(int(extent))
        return tuple(result)

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Shape addressable by this MemSpan from one runtime argument.

        Canonical-global and replicated-local storage expose the logical
        tensor. Compact storage exposes one placement-owner component even
        when its PhysicalBuffer backs all owner components contiguously.
        """

        if self.distributed_type is None or self.distributed_storage_kind.exposes_logical_coordinates:
            return self.shape
        storage_type = self.storage_distributed_type
        assert storage_type is not None
        result = []
        for value in local_shape(storage_type):
            extent = value.fixed_value if value.is_fixed else value.maximum
            if extent is None:
                raise ValueError(
                    f"BufferDescriptor {self.id!r} has an unbounded storage component shape."
                )
            result.append(int(extent))
        return tuple(result)

    @property
    def storage_distributed_type(self) -> DistributedType | None:
        """Distribution whose dense component is held by the physical pointer.

        Most buffers store the local component of their own semantic type. A
        refined shard view can instead address a strict subregion of a coarser
        same-owner component; that parent distribution is serialized here so
        editable checkpoints never rely on hidden lowering state.
        """

        return self.distributed_backing_type or self.distributed_type

    @property
    def nbytes(self) -> int:
        return self.mem_span.nbytes

    @property
    def physical_access_span(self) -> MemSpan:
        """Byte footprint of all owners in this physical allocation.

        ``mem_span`` is the pointer-visible component, not the union accessed
        by a distributed kernel. Compact-per-owner storage appends one such
        component per placement owner in the same chip-visible allocation.
        Block-local pools instead replicate the allocation itself and must
        not multiply its component by the placement size.
        """

        if self.distributed_storage_kind is not DistributedBufferStorageKind.COMPACT_PER_OWNER:
            return self.mem_span
        assert self.distributed_type is not None
        return MemSpan(
            self.mem_span.buffer,
            self.mem_span.start,
            self.mem_span.size * placement_owner_count(self.distributed_type),
        )

    @property
    def offset(self) -> int:
        return self.mem_span.offset

    @property
    def physical_id(self) -> str:
        return self.mem_span.buffer.id

    @property
    def byte_offset(self) -> int:
        return self.mem_span.byte_offset

    @property
    def alias_of(self) -> str | None:
        return None if self.alias is None else self.alias.source

    def subview(
        self,
        id: str,
        *,
        dtype: DataType,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        byte_offset: DimensionLike,
        byte_size: DimensionLike,
        alignment: int | None = None,
        source_node: str | None = None,
        field: str | None = None,
        role: str = "view",
        offset_bindings: tuple[tuple[str, str], ...] = (),
    ) -> BufferDescriptor:
        """Create a typed logical view without creating physical storage."""

        return BufferDescriptor(
            id=id,
            dtype=dtype,
            shape=tuple(shape),
            strides=tuple(strides),
            storage=self.storage,
            alignment=self.alignment if alignment is None else int(alignment),
            mem_span=self.mem_span.subspan(byte_offset, byte_size),
            source_node=self.source_node if source_node is None else source_node,
            field=field,
            alias=AliasInfo(self.id, AliasKind.VIEW),
            weight_key=self.weight_key,
            rdata_group=self.rdata_group,
            group_index=self.group_index,
            group_count=self.group_count,
            live_start=self.live_start,
            live_end=self.live_end,
            function=self.function,
            role=role,
            offset_bindings=tuple(dict((*self.offset_bindings, *offset_bindings)).items()),
            distributed_type=(
                self.distributed_type
                if tuple(shape) == self.shape
                else None
            ),
            distributed_storage_kind=(
                self.distributed_storage_kind
                if tuple(shape) == self.shape
                else DistributedBufferStorageKind.COMPACT_LOCAL
            ),
            distributed_backing_type=(
                self.distributed_backing_type
                if tuple(shape) == self.shape
                else None
            ),
        )

    def to_data(self) -> dict[str, object]:
        return {
            "id": self.id,
            "dtype": data_type_to_data(self.dtype),
            "shape": list(self.shape),
            "strides": list(self.strides),
            "storage": self.storage,
            "alignment": self.alignment,
            "mem_span": self.mem_span.to_data(),
            "source_node": self.source_node,
            "field": self.field,
            "alias": None if self.alias is None else self.alias.to_data(),
            "weight_key": self.weight_key,
            "rdata_group": self.rdata_group,
            "group_index": self.group_index,
            "group_count": self.group_count,
            "live_start": self.live_start,
            "live_end": self.live_end,
            "function": self.function,
            "role": self.role,
            **({"offset_bindings": dict(self.offset_bindings)} if self.offset_bindings else {}),
            "distributed_type": (
                None if self.distributed_type is None else self.distributed_type.to_data()
            ),
            "distributed_storage_kind": self.distributed_storage_kind.value,
            **(
                {}
                if self.distributed_backing_type is None
                else {
                    "distributed_backing_type": self.distributed_backing_type.to_data()
                }
            ),
        }

    @classmethod
    def from_data(
        cls,
        data: Mapping[str, Any],
        physical_buffers: Mapping[str, PhysicalBuffer],
    ) -> BufferDescriptor:
        return cls(
            id=str(data["id"]),
            offset_bindings=tuple(data.get("offset_bindings", {}).items()),
            dtype=data_type_from_data(data["dtype"]),
            shape=tuple(int(value) for value in data["shape"]),
            strides=tuple(int(value) for value in data["strides"]),
            storage=str(data["storage"]),
            alignment=int(data["alignment"]),
            mem_span=MemSpan.from_data(data["mem_span"], physical_buffers),
            source_node=None if data.get("source_node") is None else str(data["source_node"]),
            field=None if data.get("field") is None else str(data["field"]),
            alias=None if data.get("alias") is None else AliasInfo.from_data(data["alias"]),
            weight_key=None if data.get("weight_key") is None else str(data["weight_key"]),
            rdata_group=None if data.get("rdata_group") is None else str(data["rdata_group"]),
            group_index=None if data.get("group_index") is None else int(data["group_index"]),
            group_count=None if data.get("group_count") is None else int(data["group_count"]),
            live_start=None if data.get("live_start") is None else int(data["live_start"]),
            live_end=None if data.get("live_end") is None else int(data["live_end"]),
            function=None if data.get("function") is None else str(data["function"]),
            role=str(data.get("role", "value")),
            distributed_type=(
                None
                if data.get("distributed_type") is None
                else _require_distributed(type_from_data(data["distributed_type"]))
            ),
            distributed_storage_kind=DistributedBufferStorageKind(
                str(data.get("distributed_storage_kind", "compact_local"))
            ),
            distributed_backing_type=(
                None
                if data.get("distributed_backing_type") is None
                else _require_distributed(
                    type_from_data(data["distributed_backing_type"])
                )
            ),
        )


def _require_distributed(value) -> DistributedType:
    if not isinstance(value, DistributedType):
        raise ValueError("BufferDescriptor distributed_type must decode to DistributedType.")
    return value


__all__ = ["BufferDescriptor", "PhysicalAllocation", "PhysicalBuffer"]
