# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical compact-owner reads for a proven post-distribution QKV region."""

from collections.abc import Mapping
from math import prod

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, TupleType, VectorType, local_shard_descriptor, type_from_data
from .dimension_expression import emit_dimension
from .physical_access import emit_local_scalar_offset, emit_storage_pointer


def gather_reduce_qkv_statistics_call(raw):
    from .kernel_call_renderers import _buffer, _encode_qkv_rope_with_cache

    partials = tuple(_buffer(raw, "inputs", "qkv", index) for index in range(3))
    logical = raw["semantic_attrs"]["logical_qkv_type"]
    if isinstance(logical, Mapping):
        logical = type_from_data(logical)
    if not isinstance(logical, TupleType) or len(logical.fields) != 3:
        raise CodegenError("Fused QKV statistics require three typed logical views.")
    views = tuple({**dict(source), "abi": _logical_view_abi(source["abi"], field)}
                  for source, field in zip(partials, logical.fields, strict=True))
    return _encode_qkv_rope_with_cache(raw, *views, partial_sources=partials)


def _dense_strides(shape):
    return tuple(prod(shape[index + 1:]) for index in range(len(shape)))


def _logical_view_abi(source, value_type):
    if not isinstance(value_type, DistributedType):
        raise CodegenError("Fused QKV statistics require distributed logical views.")
    tensor = value_type.tensor
    shape = tuple(dim.fixed_value for dim in tensor.shape)
    lanes = prod(tensor.dtype.lanes) if isinstance(tensor.dtype, VectorType) else 1
    if source["dtype"] != tensor.dtype.value or int(source.get("scalar_lane_count", 1)) != lanes:
        raise CodegenError("Fused QKV view cannot change its backing dtype or lanes.")
    coordinates = tuple(f"shard_coord_{axis}" for axis in range(value_type.placement.rank))
    shard = local_shard_descriptor(value_type, coordinates)
    capacity = tuple(dim.fixed_value for dim in shard.local_capacity_shape)
    strides = _dense_strides(capacity)
    return {
        **dict(source), "logical_shape": shape, "local_capacity_shape": capacity,
        "active_shape_expressions": tuple(emit_dimension(dim) for dim in shard.active_shape),
        "logical_coordinate_expressions": tuple(
            emit_dimension(axis.map_local_to_global(f"local_coord_{index}"))
            for index, axis in enumerate(shard.axes)),
        "storage_strides": strides, "scalar_storage_strides": tuple(stride * lanes for stride in strides),
        "coordinate_space": "local", "distributed_type": value_type.to_data(),
    }


def partial_qkv_access(source, domain, layout, member):
    from .kernel_call_renderers import (
        _compact_source_coordinates, _linear_owner_expression, _static_shape, _triton_dtype,
    )

    abi = source["abi"]
    distributed = abi.get("distributed_type", {})
    partial = distributed.get("partial")
    if (abi.get("storage_kind") != "compact_per_owner" or not isinstance(partial, Mapping)
            or partial.get("reduce_op") != "sum" or not partial.get("axes")):
        raise CodegenError("Fused QKV requires compact Sum-partial owner storage.")
    hierarchy = tuple(distributed["placement"]["hierarchy"])
    axes = tuple(partial["axes"])
    shape = _static_shape(abi, "logical_shape")
    lanes = int(abi.get("scalar_lane_count", 1))
    linear = "0"
    for kind in layout:
        linear = f"(({linear}) * {domain['global_extents'][kind]} + ({domain['global_by_kind'][kind]}))"
    physical = f"(({linear}) // {lanes})"
    coordinates = tuple(f"((({physical}) // {stride}) % {extent})"
                        for extent, stride in zip(shape, _dense_strides(shape)))
    local, split_owner = _compact_source_coordinates(abi, coordinates)
    # Split-owner coordinates come from the SOURCE policies, which can differ
    # from the consumer's head ownership. Partial axes fill only the remaining
    # owner digits. Using the consumer shard index here would read wrong heads.
    partial_coordinates = ["0"] * len(hierarchy)
    remaining = member
    for axis in reversed(axes):
        partial_coordinates[axis] = f"(({remaining}) % {hierarchy[axis]})"
        remaining = f"(({remaining}) // {hierarchy[axis]})"
    owner = f"(({split_owner}) + ({_linear_owner_expression(partial_coordinates, hierarchy)}))"
    stride = int(abi.get("component_stride_scalar_elements", 0))
    if stride <= 0:
        raise CodegenError("Fused QKV partial owners need disjoint storage.")
    count = prod(hierarchy[axis] for axis in axes)
    return {
        "storage": emit_storage_pointer(abi, str(source["runtime_argument"])),
        "owner": owner, "owner_stride": stride,
        "offset": emit_local_scalar_offset(abi, local, lane_coordinate=f"(({linear}) % {lanes})"),
        "member": member, "member_count": count, "member_width": 1 << (count - 1).bit_length(),
        "scalar_dtype": _triton_dtype(str(abi["scalar_dtype"])),
    }
