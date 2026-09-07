# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral local-shard ABI used by Triton source renderers."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import (
    BufferDescriptor,
    DistributedBufferStorageKind,
    KernelExecutionKind,
    kernel_execution_kind,
    local_shard_descriptor,
    unravel_placement_index,
)


@dataclass(frozen=True)
class DistributedBufferABI:
    """Physical view of one buffer from one placement owner."""

    buffer_id: str
    storage_kind: DistributedBufferStorageKind
    owner_index: int
    owner_coordinates: tuple[int, ...]
    logical_shape: tuple[int, ...]
    local_capacity_shape: tuple[int, ...]
    active_shape: tuple[int, ...]
    component_offset_elements: int
    partial_axes: tuple[int, ...]
    partial_group_owner_indices: tuple[int, ...]
    _descriptor: object
    _storage_descriptor: object

    def logical_axis_coordinate(self, axis: int, local_coordinate: int) -> int:
        """Return the logical tensor coordinate represented by a local index."""

        try:
            value = self._descriptor.axes[axis].map_local_to_global(local_coordinate)
        except IndexError as error:
            raise CodegenError(
                f"Buffer {self.buffer_id!r} has no logical axis {axis}."
            ) from error
        if not value.is_fixed:
            raise CodegenError(
                f"Buffer {self.buffer_id!r} local coordinate did not simplify "
                "after binding a fixed owner."
            )
        return value.fixed_value

    def storage_axis_coordinate(self, axis: int, local_coordinate: int) -> int:
        """Return the coordinate used by this buffer's physical pointer."""

        if self.storage_kind.exposes_logical_coordinates:
            return self.logical_axis_coordinate(axis, local_coordinate)
        if self._storage_descriptor != self._descriptor:
            logical = self.logical_axis_coordinate(axis, local_coordinate)
            origin = self._storage_descriptor.axes[axis].map_local_to_global(0)
            if not origin.is_fixed:
                raise CodegenError(
                    f"Buffer {self.buffer_id!r} parent shard origin did not "
                    "simplify after binding a fixed owner."
                )
            return logical - origin.fixed_value
        return int(local_coordinate)


def distributed_buffer_abi(
    buffer: BufferDescriptor,
    owner_index: int,
) -> DistributedBufferABI:
    """Bind a bufferized distributed value to a placement owner.

    Compact buffers expose dense local coordinates. Canonical-global and
    replicated-local buffers retain the staged local-to-global map.
    ``COMPACT_PER_OWNER`` additionally advances the base pointer by the
    uniform component capacity; inactive tail elements remain padding and are
    masked using ``active_shape``.
    """

    distributed_type = buffer.distributed_type
    if distributed_type is None:
        raise CodegenError(
            f"Buffer {buffer.id!r} has no DistributedType local-shard ABI."
        )
    hierarchy = distributed_type.placement.hierarchy
    coordinates = unravel_placement_index(owner_index, hierarchy)
    descriptor = local_shard_descriptor(distributed_type, coordinates)
    storage_type = buffer.distributed_backing_type or distributed_type
    storage_descriptor = local_shard_descriptor(storage_type, coordinates)

    def fixed_shape(values, name):
        result = []
        for value in values:
            if not value.is_fixed:
                raise CodegenError(
                    f"Buffer {buffer.id!r} has dynamic {name} {value}; bind "
                    "runtime dimensions before requesting a static ABI."
                )
            result.append(value.fixed_value)
        return tuple(result)

    logical_shape = fixed_shape(distributed_type.tensor.shape, "logical shape")
    capacity_shape = fixed_shape(
        descriptor.local_capacity_shape, "local capacity"
    )
    active_shape = fixed_shape(descriptor.active_shape, "active shape")
    storage_capacity_shape = fixed_shape(
        storage_descriptor.local_capacity_shape, "storage local capacity"
    )
    component_elements = prod(storage_capacity_shape, start=1)
    component_offset = (
        owner_index * component_elements
        if buffer.distributed_storage_kind
        is DistributedBufferStorageKind.COMPACT_PER_OWNER
        else 0
    )
    group_coordinates = descriptor.partial_group_coordinates or ()
    group_indices = tuple(
        _linear_owner_index(coordinate, hierarchy)
        for coordinate in group_coordinates
    )
    return DistributedBufferABI(
        buffer.id,
        buffer.distributed_storage_kind,
        int(owner_index),
        coordinates,
        logical_shape,
        capacity_shape,
        active_shape,
        component_offset,
        descriptor.partial_axes,
        group_indices,
        descriptor,
        storage_descriptor,
    )


def _linear_owner_index(
    coordinates: tuple[int, ...],
    hierarchy: tuple[int, ...],
) -> int:
    result = 0
    for coordinate, extent in zip(coordinates, hierarchy):
        result = result * extent + coordinate
    return result


__all__ = [
    "DistributedBufferABI",
    "KernelExecutionKind",
    "distributed_buffer_abi",
    "kernel_execution_kind",
]
