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
    """Build one tensor map per resident owner of a packed tensor.

    The table rebases each descriptor and encodes proven affine axis strides.
    Kernels use dense-local offsets without interpreting mesh coordinates or
    split policies. A non-affine owner requires a different transfer contract.

    Which owners are resident follows the source storage. Canonical-global
    storage holds every placement owner, so the table spans the whole tensor and
    each entry rebases to that owner's origin inside it. Chip-local storage holds
    only the owners of this chip's segment, already dense and contiguous, so the
    table spans one segment and each entry names one component. Requiring the
    canonical form for every table would force a chip to hold the entire weight,
    which is the replication that chip-local storage exists to avoid.
    """

    coordinate_space = str(abi.get("coordinate_space"))
    storage_kind = str(abi.get("storage_kind"))
    if storage_kind in {"compact_per_chip", "compact_per_owner"}:
        return _compact_owner_tensor_map_table_request(
            abi,
            parameter=parameter,
            source=source,
            offset_bytes=offset_bytes,
            descriptor_shape=descriptor_shape,
            descriptor_strides=descriptor_strides,
            block_shape=block_shape,
        )
    if coordinate_space != "canonical_global":
        raise CodegenError(
            "Tensor-map tables require canonical-global or compact-owner source "
            f"storage, not {coordinate_space!r}."
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
        local_descriptor_shape = tuple(owner_extents) + physical_shape[len(logical_shape):]
        if 0 in owner_extents:
            # Tensor maps cannot encode zero extents. An empty owner's unused
            # entry names one in-bounds logical element, retaining packed lanes;
            # its actual zero active domain remains in the kernel's shard ABI.
            base_scalar_elements = 0
            local_descriptor_shape = (1,) * len(logical_shape) + physical_shape[len(logical_shape):]
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


def _compact_owner_tensor_map_table_request(
    abi: Mapping[str, object],
    *,
    parameter: str,
    source: str,
    offset_bytes: int,
    descriptor_shape: Sequence[int],
    descriptor_strides: Sequence[int],
    block_shape: Sequence[int],
) -> dict[str, object]:
    """Build one tensor map per owner component of compact-owner storage.

    Compact-owner storage is already the per-owner projection: each resident
    owner occupies one dense component of ``component_stride`` scalar elements,
    and the descriptor geometry describes that component rather than the
    canonical tensor. Owner selection is therefore pure component indexing, with
    no mesh coordinates and no split policy to interpret; the kernel indexes the
    table by the same owner expression the buffer ABI already publishes.
    """

    physical_shape = _positive_int_tuple(descriptor_shape, "descriptor shape")
    physical_strides = _positive_int_tuple(
        descriptor_strides, "descriptor strides"
    )
    tile_shape = _positive_int_tuple(block_shape, "descriptor block shape")
    local_shape = _positive_int_tuple(
        abi.get("local_capacity_shape"), "local capacity shape"
    )
    if not (
        len(physical_shape) == len(physical_strides) == len(tile_shape)
        and physical_shape[:len(local_shape)] == local_shape
    ):
        raise CodegenError(
            "Compact-owner tensor-map table descriptor does not preserve its "
            "local capacity prefix."
        )
    scalar_dtype = str(abi.get("scalar_dtype", ""))
    try:
        item_size = _DTYPE_ITEM_SIZES[scalar_dtype]
    except KeyError as error:
        raise CodegenError(
            f"Compact-owner tensor-map table does not support dtype {scalar_dtype!r}."
        ) from error
    if offset_bytes < 0:
        raise CodegenError(
            "Compact-owner tensor-map table source offset must be non-negative."
        )
    owner_count = int(abi.get("owner_count", 0))
    if owner_count <= 0:
        raise CodegenError(
            "Compact-owner tensor-map table requires resident owners."
        )
    component_stride = int(abi.get("component_stride_scalar_elements", 0))
    if component_stride <= 0:
        raise CodegenError(
            "Compact-owner tensor-map table source has no scalar owner stride."
        )
    component_span = 1 + sum(
        (extent - 1) * stride
        for extent, stride in zip(physical_shape, physical_strides, strict=True)
    )
    if component_stride < component_span:
        raise CodegenError(
            "Compact-owner tensor-map payloads overlap in physical storage."
        )
    return {
        "parameter": parameter,
        "source": source,
        "kind": "table",
        "dtype": scalar_dtype,
        "block_shape": tile_shape,
        "padding": "zero",
        "swizzle_mode": _tma_swizzle_mode(tile_shape, item_size),
        "entry_size_bytes": 128,
        "entries": tuple(
            {
                "offset_bytes": offset_bytes + owner * component_stride * item_size,
                "shape": physical_shape,
                "strides": physical_strides,
                "source_shape_axes": tuple(() for _ in physical_shape),
            }
            for owner in range(owner_count)
        ),
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
