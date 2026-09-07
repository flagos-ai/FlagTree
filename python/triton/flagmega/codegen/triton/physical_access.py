# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Emit structured local-shard accesses for Triton package renderers.

This is the renderer-side equivalent of nncase/PyNTT's canonical access
rewriter. Kernel algorithms are written over dense local coordinates. The
buffer ABI decides whether those coordinates address a compact owner slice or
must first be mapped into canonical-global storage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from triton.flagmega.errors import CodegenError


_TRITON_SCALAR_TYPES = {
    "bool": "tl.int1",
    "int8": "tl.int8",
    "uint8": "tl.uint8",
    "int16": "tl.int16",
    "uint16": "tl.uint16",
    "int32": "tl.int32",
    "uint32": "tl.uint32",
    "int64": "tl.int64",
    "uint64": "tl.uint64",
    "float16": "tl.float16",
    "bfloat16": "tl.bfloat16",
    "float32": "tl.float32",
    "float64": "tl.float64",
    "float8_e4m3fn": "tl.float8e4nv",
}


def emit_triton_scalar_type(dtype: str) -> str:
    """Return Triton's spelling for one serialized scalar dtype."""

    try:
        return _TRITON_SCALAR_TYPES[str(dtype)]
    except KeyError as error:
        raise CodegenError(
            f"Triton has no scalar spelling for dtype {dtype!r}."
        ) from error


def emit_buffer_pointer(
    abi: Mapping[str, object],
    argument: str,
    *,
    owner_index: str = "shard_index",
) -> str:
    """Return a typed scalar pointer for one local-buffer ABI binding."""

    pointer = emit_storage_pointer(abi, argument)
    return emit_owner_base(abi, pointer, owner_index=owner_index)


def emit_storage_pointer(
    abi: Mapping[str, object],
    argument: str,
) -> str:
    """Return the allocation/component-table base before owner selection."""

    storage = str(abi.get("storage"))
    if storage == "scalar":
        raise CodegenError("Compile-time/runtime scalar values are not buffer pointers.")
    if storage in {"rdata", "workspace"} or bool(abi.get("pooled", False)):
        scalar_dtype = _TRITON_SCALAR_TYPES.get(str(abi.get("scalar_dtype")))
        if scalar_dtype is None:
            raise CodegenError(
                "Triton pooled storage cannot cast scalar dtype "
                f"{abi.get('scalar_dtype')!r}."
            )
        itemsize = int(abi["scalar_itemsize"])
        byte_offset = int(abi["pool_byte_offset"])
        scope_stride = int(abi.get("pool_scope_stride_bytes", 0))
        scope_index = abi.get("pool_scope_index")
        if itemsize <= 0 or byte_offset % itemsize or scope_stride % itemsize:
            raise CodegenError(
                f"Pooled buffer offset/stride {byte_offset}/{scope_stride} is not aligned to "
                f"{itemsize}-byte {abi.get('scalar_dtype')} elements."
            )
        pointer = f"{argument}.to(tl.pointer_type({scalar_dtype}))"
        terms = []
        if scope_stride:
            if not isinstance(scope_index, str) or not scope_index:
                raise CodegenError("Replicated pooled storage has no scope index.")
            terms.append(f"({scope_index}) * {scope_stride // itemsize}")
        if byte_offset:
            terms.append(str(byte_offset // itemsize))
        if terms:
            pointer = f"({pointer} + {' + '.join(terms)})"
    else:
        pointer = str(argument)
    return pointer


def emit_scalar_immediate(
    abi: Mapping[str, object],
    value: str,
) -> str:
    """Materialize a scalar constant as a runtime rank-zero SSA value.

    A Python literal crossing a reusable Triton function boundary becomes a
    ``constexpr`` specialization parameter.  Repeated calls with different
    literals would therefore clone the same function body.  ``tl.full`` keeps
    the scalar value dynamic at that boundary and preserves code reuse.
    """

    if str(abi.get("storage")) != "scalar":
        raise CodegenError("Only scalar ABI values can materialize an immediate.")
    scalar_dtype = _TRITON_SCALAR_TYPES.get(str(abi.get("scalar_dtype")))
    if scalar_dtype is None:
        raise CodegenError(
            "Triton cannot materialize scalar dtype "
            f"{abi.get('scalar_dtype')!r}."
        )
    return f"tl.full((), {value}, {scalar_dtype})"


def emit_owner_base(
    abi: Mapping[str, object],
    argument: str,
    *,
    owner_index: str = "shard_index",
) -> str:
    """Return the scalar-pointer base for the current owner's component."""

    stride = int(abi.get("component_stride_scalar_elements", 0))
    if str(abi.get("storage_kind")) == "compact_per_owner":
        if stride <= 0:
            raise CodegenError("Compact-per-owner buffer has no scalar owner stride.")
        return f"({argument} + ({owner_index}) * {stride})"
    return str(argument)


def emit_local_scalar_offset(
    abi: Mapping[str, object],
    local_coordinates: Sequence[str],
    *,
    lane_coordinate: str | None = None,
    shard_coordinates: Sequence[str] | None = None,
) -> str:
    """Map dense local coordinates to a scalar-element storage offset."""

    shard_coordinates = _shard_coordinates(abi, shard_coordinates)
    strides = tuple(int(value) for value in abi["scalar_storage_strides"])
    if len(local_coordinates) != len(strides):
        raise CodegenError(
            "Local access rank does not match the physical buffer ABI: "
            f"{len(local_coordinates)} != {len(strides)}."
        )
    coordinate_space = str(abi.get("coordinate_space"))
    if coordinate_space in {"canonical_global", "parent_shard_local"}:
        expression_key = (
            "logical_coordinate_expressions"
            if coordinate_space == "canonical_global"
            else "storage_coordinate_expressions"
        )
        raw_coordinates = tuple(str(value) for value in abi[expression_key])
        if len(raw_coordinates) != len(strides):
            raise CodegenError(
                f"{coordinate_space} buffer coordinate rank is incomplete."
            )
        coordinates = tuple(
            _bind_coordinate_expression(
                expression,
                local_coordinates,
                shard_coordinates,
            )
            for expression in raw_coordinates
        )
    else:
        coordinates = tuple(str(value) for value in local_coordinates)
    terms = [
        f"({coordinate}) * {stride}"
        for coordinate, stride in zip(coordinates, strides, strict=True)
        if stride
    ]
    if lane_coordinate is not None:
        lane_count = int(abi.get("scalar_lane_count", 1))
        if lane_count == 1:
            raise CodegenError("A scalar buffer cannot be indexed with a vector lane.")
        terms.append(f"({lane_coordinate})")
    return " + ".join(terms) if terms else "0"


def emit_logical_coordinate(
    abi: Mapping[str, object],
    axis: int,
    local_coordinates: Sequence[str],
    *,
    shard_coordinates: Sequence[str] | None = None,
) -> str:
    """Map one dense local tensor coordinate to its logical coordinate."""

    shard_coordinates = _shard_coordinates(abi, shard_coordinates)
    expressions = tuple(
        str(value) for value in abi["logical_coordinate_expressions"]
    )
    if len(local_coordinates) != len(expressions):
        raise CodegenError(
            "Logical-coordinate rank does not match the physical buffer ABI."
        )
    try:
        expression = expressions[axis]
    except IndexError as error:
        raise CodegenError(
            f"Logical-coordinate axis {axis} is outside rank {len(expressions)}."
        ) from error
    return _bind_coordinate_expression(
        expression, local_coordinates, shard_coordinates
    )


def emit_global_scalar_offset(
    abi: Mapping[str, object],
    logical_coordinates: Sequence[str],
    *,
    lane_coordinate: str | None = None,
) -> str:
    """Address canonical storage from already-global logical coordinates."""

    strides = tuple(int(value) for value in abi["scalar_storage_strides"])
    if len(logical_coordinates) != len(strides):
        raise CodegenError(
            "Global access rank does not match the physical buffer ABI."
        )
    terms = [
        f"({coordinate}) * {stride}"
        for coordinate, stride in zip(logical_coordinates, strides, strict=True)
        if stride
    ]
    if lane_coordinate is not None:
        lane_count = int(abi.get("scalar_lane_count", 1))
        if lane_count == 1:
            raise CodegenError("A scalar buffer cannot be indexed with a vector lane.")
        terms.append(f"({lane_coordinate})")
    return " + ".join(terms) if terms else "0"


def emit_active_extent(
    abi: Mapping[str, object],
    axis: int,
    *,
    shard_coordinates: Sequence[str] | None = None,
) -> str:
    shard_coordinates = _shard_coordinates(abi, shard_coordinates)
    expressions = tuple(str(value) for value in abi["active_shape_expressions"])
    try:
        expression = expressions[axis]
    except IndexError as error:
        raise CodegenError(
            f"Active-extent axis {axis} is outside rank {len(expressions)}."
        ) from error
    return _bind_coordinate_expression(expression, (), shard_coordinates)


def _shard_coordinates(
    abi: Mapping[str, object],
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    if explicit is not None:
        return tuple(str(value) for value in explicit)
    distributed = abi.get("distributed_type")
    if not isinstance(distributed, Mapping):
        return ("shard_y", "shard_x")
    placement = distributed.get("placement")
    if not isinstance(placement, Mapping):
        raise CodegenError("Distributed buffer ABI has no placement metadata.")
    hierarchy = placement.get("hierarchy")
    if not isinstance(hierarchy, (tuple, list)) or not hierarchy:
        raise CodegenError("Distributed buffer placement requires a hierarchy.")
    rank = len(hierarchy)
    if rank == 2:
        # Preserve the established 2-D source ABI while higher/lower ranks use
        # nncase's placement-indexed spelling.
        return ("shard_y", "shard_x")
    return tuple(f"shard_coord{axis}" for axis in range(rank))


def _bind_coordinate_expression(
    expression: str,
    local_coordinates: Sequence[str],
    shard_coordinates: Sequence[str],
) -> str:
    result = str(expression)
    # Replace longer numeric suffixes first (axis 10 before axis 1).
    for axis in reversed(range(len(local_coordinates))):
        result = result.replace(
            f"local_coord_{axis}", f"({local_coordinates[axis]})"
        )
    for axis in reversed(range(len(shard_coordinates))):
        result = result.replace(
            f"shard_coord_{axis}", f"({shard_coordinates[axis]})"
        )
    if "local_coord_" in result or "shard_coord_" in result:
        raise CodegenError(
            f"Unbound local-shard coordinate in expression {result!r}."
        )
    return result


__all__ = [
    "emit_active_extent",
    "emit_buffer_pointer",
    "emit_global_scalar_offset",
    "emit_local_scalar_offset",
    "emit_logical_coordinate",
    "emit_owner_base",
    "emit_scalar_immediate",
    "emit_storage_pointer",
    "emit_triton_scalar_type",
]
