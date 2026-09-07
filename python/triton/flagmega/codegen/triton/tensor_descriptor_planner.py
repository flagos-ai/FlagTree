# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host tensor-map plans derived from the distributed buffer ABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.distributed_type import placement_from_data, sbp_from_data
from triton.flagmega.ir.local_shard import local_shard_descriptor
from triton.flagmega.ir.model import DistributedType, tensor_type


_DTYPE_ITEM_SIZES = {
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8e4m3fn": 1,
    "float8_e5m2": 1,
    "float8e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


def packed_distributed_tensor_map_table_request(
    abi: Mapping[str, object],
    *,
    parameter: str,
    source: str,
    offset_bytes: int,
    descriptor_shape: Sequence[int],
    descriptor_strides: Sequence[int],
    block_shape: Sequence[int],
) -> dict[str, object]:
    """Build one tensor map per owner of a canonical packed tensor.

    The table rebases each descriptor and encodes proven affine axis strides.
    Kernels use dense-local offsets without interpreting mesh coordinates or
    split policies. A non-affine owner requires a different transfer contract.
    """

    if str(abi.get("coordinate_space")) != "canonical_global":
        raise CodegenError(
            "Tensor-map tables require canonical-global source storage."
        )
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        raise CodegenError(
            "Tensor-map tables require a serialized DistributedType."
        )
    placement = distributed.get("placement")
    hierarchy = (
        placement.get("hierarchy") if isinstance(placement, Mapping) else None
    )
    policies = distributed.get("axis_policies")
    logical_shape = _positive_int_tuple(abi.get("logical_shape"), "logical shape")
    local_shape = _positive_int_tuple(
        abi.get("local_capacity_shape"), "local capacity shape"
    )
    physical_shape = _positive_int_tuple(descriptor_shape, "descriptor shape")
    physical_strides = _positive_int_tuple(
        descriptor_strides, "descriptor strides"
    )
    tile_shape = _positive_int_tuple(block_shape, "descriptor block shape")
    mesh_shape = _positive_int_tuple(hierarchy, "mesh hierarchy")
    if (
        not isinstance(policies, (tuple, list))
        or len(policies) != len(logical_shape)
        or len(local_shape) != len(logical_shape)
    ):
        raise CodegenError(
            "Tensor-map table distributed policy/shape ranks differ."
        )
    if not (
        len(physical_shape) == len(physical_strides) == len(tile_shape)
        and physical_shape[:len(logical_shape)] == logical_shape
    ):
        raise CodegenError(
            "Tensor-map table physical descriptor does not preserve its logical prefix."
        )
    scalar_dtype = str(abi.get("scalar_dtype", ""))
    try:
        item_size = _DTYPE_ITEM_SIZES[scalar_dtype]
    except KeyError as error:
        raise CodegenError(
            f"Tensor-map table does not support dtype {scalar_dtype!r}."
        ) from error
    if offset_bytes < 0:
        raise CodegenError("Tensor-map table source offset must be non-negative.")

    try:
        shard_type = DistributedType(
            tensor_type("bfloat16", logical_shape),
            tuple(sbp_from_data(policy) for policy in policies),
            placement_from_data(placement),
        )
    except (IRSchemaError, KeyError, TypeError, ValueError) as error:
        raise CodegenError(f"Invalid tensor-map distributed geometry: {error}") from error

    entries = []
    owner_count = prod(mesh_shape)
    for linear_owner in range(owner_count):
        owner = _unflatten_owner(linear_owner, mesh_shape)
        owner_origins = []
        owner_extents = []
        owner_strides = list(physical_strides)
        descriptor = local_shard_descriptor(shard_type, owner)
        for axis, shard_axis in enumerate(descriptor.axes):
            stride = shard_axis.affine_stride
            if stride is None:
                raise CodegenError(f"Tensor-map logical axis {axis} is not an affine shard for owner {owner}.")
            origin = shard_axis.map_local_to_global(0).fixed_value
            active = shard_axis.active_extent.fixed_value
            capacity = shard_axis.local_capacity.fixed_value
            if capacity != local_shape[axis]:
                raise CodegenError(
                    "Tensor-map table local capacity disagrees with the buffer ABI: "
                    f"axis {axis} has {capacity}, expected {local_shape[axis]}."
                )
            owner_origins.append(origin)
            owner_extents.append(active)
            owner_strides[axis] *= stride
        base_scalar_elements = sum(
            origin * physical_strides[axis]
            for axis, origin in enumerate(owner_origins)
        )
        local_descriptor_shape = tuple(
            (
                active
                if active > 0
                else tile_shape[axis]
            )
            for axis, active in enumerate(owner_extents)
        ) + physical_shape[len(logical_shape):]
        entries.append({
            "offset_bytes": offset_bytes + base_scalar_elements * item_size,
            "shape": local_descriptor_shape,
            "strides": tuple(owner_strides),
            "source_shape_axes": tuple(() for _ in physical_shape),
        })

    return {
        "parameter": parameter,
        "source": source,
        "kind": "table",
        "dtype": scalar_dtype,
        "block_shape": tile_shape,
        "padding": "zero",
        "swizzle_mode": _tma_swizzle_mode(tile_shape, item_size),
        "entry_size_bytes": 128,
        "entries": tuple(entries),
    }


def packed_owner_prefix_tensor_map_table_request(
    abi: Mapping[str, object],
    *,
    parameter: str,
    source: str,
    offset_bytes: int,
    owner_count: int,
    descriptor_shape: Sequence[int],
    descriptor_strides: Sequence[int],
    block_shape: Sequence[int],
) -> dict[str, object]:
    """Build one tensor map per dense leading owner dimension.

    Some packed constant layouts make placement ownership an explicit leading
    tensor dimension rather than a ``DistributedType`` policy.  This is a
    physical-layout contract: every table entry drops that leading dimension
    and rebases to one non-overlapping owner payload.  The kernel therefore
    issues only owner-local coordinates, exactly as for a distributed table.
    """

    physical_shape = _positive_int_tuple(descriptor_shape, "descriptor shape")
    physical_strides = _positive_int_tuple(
        descriptor_strides, "descriptor strides"
    )
    tile_shape = _positive_int_tuple(block_shape, "descriptor block shape")
    if owner_count <= 0:
        raise CodegenError("Tensor-map owner-prefix table requires owners.")
    if not (
        len(physical_shape) == len(physical_strides) == len(tile_shape)
        and len(physical_shape) >= 2
        and physical_shape[0] == owner_count
        and tile_shape[0] == 1
    ):
        raise CodegenError(
            "Tensor-map owner-prefix table requires matching ranks, one "
            "leading owner extent, and a unit owner tile."
        )
    scalar_dtype = str(abi.get("scalar_dtype", ""))
    try:
        item_size = _DTYPE_ITEM_SIZES[scalar_dtype]
    except KeyError as error:
        raise CodegenError(
            f"Tensor-map owner-prefix table does not support dtype {scalar_dtype!r}."
        ) from error
    if offset_bytes < 0:
        raise CodegenError(
            "Tensor-map owner-prefix table source offset must be non-negative."
        )
    owner_stride = physical_strides[0]
    owner_span = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(
            physical_shape[1:], physical_strides[1:], strict=True
        )
    )
    if owner_stride < owner_span:
        raise CodegenError(
            "Tensor-map owner-prefix payloads overlap in physical storage."
        )

    local_shape = physical_shape[1:]
    local_strides = physical_strides[1:]
    local_tile = tile_shape[1:]
    return {
        "parameter": parameter,
        "source": source,
        "kind": "table",
        "dtype": scalar_dtype,
        "block_shape": local_tile,
        "padding": "zero",
        "swizzle_mode": _tma_swizzle_mode(local_tile, item_size),
        "entry_size_bytes": 128,
        "entries": tuple(
            {
                "offset_bytes": offset_bytes + owner * owner_stride * item_size,
                "shape": local_shape,
                "strides": local_strides,
                "source_shape_axes": tuple(() for _ in local_shape),
            }
            for owner in range(owner_count)
        ),
    }


def _unflatten_owner(
    linear_owner: int, hierarchy: tuple[int, ...]
) -> tuple[int, ...]:
    coordinates = [0] * len(hierarchy)
    remainder = linear_owner
    for axis in range(len(hierarchy) - 1, -1, -1):
        coordinates[axis] = remainder % hierarchy[axis]
        remainder //= hierarchy[axis]
    return tuple(coordinates)


def _positive_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in value
        )
    ):
        raise CodegenError(f"Tensor-map table {name} must contain positive integers.")
    return tuple(int(item) for item in value)


def _tma_swizzle_mode(block_shape: tuple[int, ...], item_size: int) -> int:
    if len(block_shape) < 2 or prod(block_shape[:-1]) < 8:
        return 0
    contiguous_bytes = block_shape[-1] * item_size
    return {128: 3, 64: 2, 32: 1}.get(contiguous_bytes, 0)


__all__ = [
    "packed_distributed_tensor_map_table_request",
    "packed_owner_prefix_tensor_map_table_request",
]
