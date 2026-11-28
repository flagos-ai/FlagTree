# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense local row/reduction domains mapped by the physical buffer ABI."""

from math import prod

from triton.flagmega.codegen.triton.physical_access import emit_active_extent
from triton.flagmega.errors import CodegenError


def local_reduction_domain(abi, axes, maximum_tile, *, row_index="_fm_row", reduction_index="_fm_offsets",
                           lane_wise=False):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _unflattened_coordinates

    shape = tuple(int(dim) for dim in abi["local_capacity_shape"])
    lanes = int(abi.get("scalar_lane_count", 1))
    if lanes != 1 and not lane_wise:
        raise CodegenError("Scalar axis reductions do not accept vector element types.")
    if lanes <= 0:
        raise CodegenError("A reduction requires a positive scalar lane count.")
    if maximum_tile <= 0 or maximum_tile & (maximum_tile - 1):
        raise CodegenError("A reduction tile must be a positive power of two.")
    outer_axes = tuple(axis for axis in range(len(shape)) if axis not in axes)
    # Empty axes still have a valid (never dereferenced) coordinate expression.
    # Do not generate modulo/division by zero in dead masked lanes.
    # Runtime argument names are sanitized without leading underscores. Keep
    # algorithm temporaries in that separate namespace, including coordinates.
    outer_index = f"(({row_index}) // {lanes})" if lanes != 1 else row_index
    outer_coordinates = _unflattened_coordinates(tuple(max(shape[axis], 1) for axis in outer_axes), outer_index)
    reduce_coordinates = _unflattened_coordinates(tuple(max(shape[axis], 1) for axis in axes), reduction_index)
    coordinates = dict(zip(outer_axes, outer_coordinates, strict=True))
    coordinates.update(zip(axes, reduce_coordinates, strict=True))

    def active(selected):
        return " & ".join(f"(({coordinates[axis]}) < ({emit_active_extent(abi, axis)}))" for axis in selected) or "True"

    capacity = prod(shape[axis] for axis in axes)
    return {
        "rows": prod(shape[axis] for axis in outer_axes) * lanes,
        "capacity": capacity,
        "tile": min(maximum_tile, 1 << (max(capacity, 1) - 1).bit_length()),
        "coordinates": tuple(coordinates[axis] for axis in range(len(shape))),
        "outer_coordinates": outer_coordinates,
        "outer_active": active(outer_axes),
        "active": active(range(len(shape))),
        **({"lane_coordinate": f"(({row_index}) % {lanes})"} if lanes != 1 else {}),
    }
