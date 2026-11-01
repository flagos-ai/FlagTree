# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed owner domain for independent FP8 scale-block projections on a chip."""

from dataclasses import dataclass
from math import prod

from triton.flagmega.ir import (
    SBP,
    DistributedType,
    VectorType,
    local_tensor_type,
    logical_type,
    scale_split_units,
)
from triton.flagmega.ir.distributed_type import ContiguousSplit, SBPSplit


@dataclass(frozen=True)
class CooperativeGateGeometry:
    tokens: int
    local_routes: int
    local_rows: int
    reduction: int
    hierarchy: tuple[int, ...]
    levels: str
    route_axes: tuple[int, ...]
    row_axes: tuple[int, ...]

    @property
    def chip_routes(self):
        return self.local_routes * prod(self.hierarchy[i] for i in self.route_axes)

    @property
    def chip_rows(self):
        return self.local_rows * prod(self.hierarchy[i] for i in self.row_axes)

    @property
    def row_tiles(self):
        return -(-self.chip_rows // 64)

    @property
    def k_blocks(self):
        return self.reduction // 128

    @property
    def partial_shape(self):
        return (self.tokens, self.chip_routes, 2, self.row_tiles, self.k_blocks, 64)

    @property
    def scale_shape(self):
        return (self.tokens, self.chip_routes, self.k_blocks)


def cooperative_gate_geometry(types, result_type):
    """Return a complete fixed owner domain, or reject an unproved mapping.

    A task spans scale blocks, route owners and row owners *within* a chip.
    Other placement coordinates stay fixed. Contiguous owner stages may have
    tails and arbitrary granularity; their existing local coordinate maps
    remain authoritative when the renderer addresses a selected owner.
    """
    values = (*types.values(), result_type)
    distributed = tuple(t for t in values if isinstance(t, DistributedType))
    if any(t.partial is not None or t.exclusive is not None for t in distributed):
        return None
    placement = distributed[0].placement if distributed else None
    if any(t.placement != placement for t in distributed):
        return None
    if any(not extent.is_fixed for t in values for extent in logical_type(t).shape):
        return None

    def policy(value, axis):
        return value.axis_policies[axis] if isinstance(value, DistributedType) else SBP.broadcast()

    def local(value):
        return local_tensor_type(value) if isinstance(value, DistributedType) else value

    def contiguous(value):
        return value == SBP.broadcast() or (
            isinstance(value, SBPSplit)
            and len(value.stages) == 1
            and isinstance(value.stages[0].distribution, ContiguousSplit)
        )

    activation, ids = types["dispatched"], types["router_expert_ids"]
    gate, up = types["gate_weight"], types["up_weight"]
    output = logical_type(result_type)
    lanes = output.dtype.lane_count if isinstance(output.dtype, VectorType) else 1
    if (
        logical_type(activation).rank != 3
        or logical_type(ids).rank != 2
        or logical_type(gate).rank != 3
        or logical_type(up).rank != 3
        or output.rank != 3
    ):
        return None
    route = policy(ids, 1)
    row = policy(gate, 1)
    output_row = policy(result_type, 2)
    if isinstance(output_row, SBPSplit):
        output_row = scale_split_units(output_row, lanes, 1)
    if (
        not contiguous(route)
        or not contiguous(row)
        or policy(activation, 1) != route
        or policy(result_type, 1) != route
        or policy(up, 1) != row
        or output_row != row
    ):
        return None
    for value, axes in ((activation, (0, 2)), (ids, (0,)), (gate, (0, 2)), (up, (0, 2)), (result_type, (0,))):
        if any(policy(value, axis) != SBP.broadcast() for axis in axes):
            return None
    for name, value in types.items():
        if name.endswith("_scale") and any(
            policy(value, i) != SBP.broadcast() for i in range(logical_type(value).rank)
        ):
            return None
    hierarchy = tuple(placement.hierarchy) if placement else ()
    levels = placement.hierarchy_levels if placement else ""
    route_axes = tuple(i for i in getattr(route, "hierarchy_axes", ()) if levels[i] == "b")
    row_axes = tuple(i for i in getattr(row, "hierarchy_axes", ()) if levels[i] == "b")
    if set(route_axes) & set(row_axes):
        return None
    geometry = CooperativeGateGeometry(
        local(result_type).shape[0].fixed_value,
        local(ids).shape[1].fixed_value,
        local(gate).shape[1].fixed_value,
        local(gate).shape[2].fixed_value,
        hierarchy,
        levels,
        route_axes,
        row_axes,
    )
    if min(geometry.tokens, geometry.local_routes, geometry.local_rows, geometry.reduction) <= 0:
        return None
    if geometry.reduction % 128:
        return None
    return geometry
