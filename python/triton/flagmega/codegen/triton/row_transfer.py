# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Proof obligations for an unmasked owner-local asynchronous row copy."""

from itertools import product
from math import prod

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, TensorType, local_shard_descriptor, logical_type, type_from_data


def has_full_static_rows(value_type) -> bool:
    """Every owner must have one complete row, including boundary owners."""
    value = logical_type(value_type)
    if not isinstance(value, TensorType) or not value.shape or not all(dim.is_fixed for dim in value.shape):
        return False
    if isinstance(value_type, DistributedType):
        for owner in product(*(range(n) for n in value_type.placement.hierarchy)):
            shard = local_shard_descriptor(value_type, owner)
            if (any(not axis.active_extent.is_fixed or not axis.local_capacity.is_fixed
                    or axis.active_extent.fixed_value != axis.local_capacity.fixed_value for axis in shard.axes)
                    or prod(axis.local_capacity.fixed_value for axis in shard.axes[:-1]) != 1
                    or shard.axes[-1].local_capacity.fixed_value <= 0):
                return False
        return True
    return prod(dim.fixed_value for dim in value.shape[:-1]) == 1 and value.shape[-1].fixed_value > 0


def validate_async_row_copy(abi, copy_tile: int) -> None:
    """Prove full tiles, contiguous physical elements, and aligned owner bases.

    No SBP spelling implies physical contiguity. Compact storage uses local
    coordinates; canonical/parent storage additionally needs an affine unit
    step and an aligned origin for every owner. External pointers lack the
    prepared runtime's owned-pool base contract and are not accepted here.
    """
    def reject(reason):
        raise CodegenError(f"LHS async copy requires {reason}.")

    shape = tuple(abi["local_capacity_shape"])
    lanes = int(abi.get("scalar_lane_count", 1))
    itemsize = int(abi["scalar_itemsize"])
    strides = tuple(abi["scalar_storage_strides"])
    if (not shape or prod(shape[:-1]) != 1 or lanes <= 0
            or shape[-1] * lanes % copy_tile or strides[-1] != lanes):
        reject("a contiguous scalar row with complete copy tiles")
    if not abi.get("pooled", False) or int(abi.get("alignment_bytes", 0)) < 16:
        reject("an owned pool with a proven 16-byte-aligned view")
    if int(abi.get("pool_byte_offset", 0)) % 16 or int(abi.get("pool_scope_stride_bytes", 0)) % 16:
        reject("16-byte-aligned pool offsets and scope strides")
    if abi.get("storage_kind") == "compact_per_owner":
        component_bytes = int(abi.get("component_stride_scalar_elements", 0)) * itemsize
        if component_bytes <= 0 or component_bytes % 16:
            reject("16-byte-aligned positive owner components")

    data = abi.get("distributed_type")
    value_type = type_from_data(data) if data is not None else None
    if value_type is not None and not has_full_static_rows(value_type):
        reject("fully active static rows for every owner")
    space = abi.get("coordinate_space")
    if space not in {"local", "canonical_global", "parent_shard_local"}:
        reject("a known physical coordinate map")
    if space == "local" or (space == "canonical_global" and value_type is None):
        return
    if value_type is None:
        reject("a distributed view for parent-local storage")
    backing = abi.get("distributed_backing_type")
    backing_type = type_from_data(backing) if backing is not None else None
    for owner in product(*(range(n) for n in value_type.placement.hierarchy)):
        shard = local_shard_descriptor(value_type, owner)
        if shard.axes[-1].affine_stride != 1:
            reject("a unit-stride physical reduction axis")
        origins = [axis.map_local_to_global(0) for axis in shard.axes]
        if space == "parent_shard_local":
            if not isinstance(backing_type, DistributedType):
                reject("the parent shard type for parent-local storage")
            parent = local_shard_descriptor(backing_type, owner)
            origins = [(origin - axis.map_local_to_global(0)).simplify()
                       for origin, axis in zip(origins, parent.axes, strict=True)]
        if any(not origin.is_fixed for origin in origins):
            reject("statically aligned physical shard origins")
        offset = sum(origin.fixed_value * stride for origin, stride in zip(origins, strides, strict=True))
        if offset * itemsize % 16:
            reject("16-byte-aligned physical shard origins")


__all__ = ["has_full_static_rows", "validate_async_row_copy"]
