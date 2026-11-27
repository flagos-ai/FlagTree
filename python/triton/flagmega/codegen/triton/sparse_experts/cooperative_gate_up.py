# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical chip-peer access and scratch ABI for ordered FP8 gate/up blocks."""

from math import prod

from triton.flagmega.codegen.triton.cooperative_access import (
    chip_peer_pointer,
    require_chip_shared,
    require_chip_workspace,
    require_owner_output,
)
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_local_scalar_offset,
    emit_logical_coordinate,
)
from triton.flagmega.codegen.triton.sparse_experts.cooperative_geometry import cooperative_gate_geometry
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError


def _linear(geometry, axes):
    result = "0"
    for axis in axes:
        result = f"({result} * {geometry.hierarchy[axis]} + shard_coord{axis})"
    return result


def _peer_coordinates(geometry, *, route="0", row="0"):
    result = [f"shard_coord{i}" if level != "b" else "0" for i, level in enumerate(geometry.levels)]
    for axes, index in ((geometry.route_axes, route), (geometry.row_axes, row)):
        for position, axis in enumerate(axes):
            stride = prod(geometry.hierarchy[i] for i in axes[position + 1 :])
            result[axis] = f"((({index}) // {stride}) % {geometry.hierarchy[axis]})"
    return tuple(result)


def _peer_pointer(binding, coordinates):
    return chip_peer_pointer(binding, coordinates, owner="Cooperative FP8 experts")


def _shared(binding, name):
    require_chip_shared(binding, name, owner="Cooperative FP8 experts")


def _workspace(binding, shape, name):
    require_chip_workspace(binding, shape, name, owner="Cooperative FP8 experts")


def cooperative_gate_up_context(raw, context, operands, result):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _exclusive_participation_active,
        _pointer,
    )

    geometry = cooperative_gate_geometry(
        {name: _tensor_type(value["abi"]) for name, value in operands.items()}, _tensor_type(result["abi"])
    )
    if geometry is None:
        raise CodegenError("Cooperative FP8 experts require a complete contiguous chip owner domain.")
    if (
        raw.get("facts", {}).get("internal_grid_barriers") != 1
        or raw.get("facts", {}).get("source_memory_scope") != "chip"
        or _exclusive_participation_active(raw) != "True"
    ):
        raise CodegenError("Cooperative FP8 experts require all CTAs, chip read effects and one internal barrier.")
    merge_n = raw["parameters"].get("merge_n")
    if type(merge_n) is not int or merge_n != 32:
        raise CodegenError("Cooperative FP8 experts require a 32-row ordered merge.")
    partials = _buffer(raw, "workspaces", "projection_partials")
    scales = _buffer(raw, "workspaces", "activation_scales")
    _workspace(partials, geometry.partial_shape, "projection_partials")
    _workspace(scales, geometry.scale_shape, "activation_scales")
    for name in ("dispatched", "router_expert_ids", "gate_weight", "up_weight", "gate_proj_scale", "up_proj_scale"):
        _shared(operands[name], name)
    require_owner_output(result, "result", owner="Cooperative FP8 experts")

    return {
        **gate_up_peer_context(geometry, context, operands),
        "internal_grid_barriers": 1,
        "partial_pointer": _pointer(partials),
        "activation_scale_pointer": _pointer(scales),
    }


def gate_up_peer_context(geometry, context, operands):
    """Address complete chip route/row domains using the physical owner maps."""
    from triton.flagmega.codegen.triton.kernel_call_renderers import _pointer

    route_coordinates = _peer_coordinates(geometry, route="_fm_route_owner")
    row_coordinates = _peer_coordinates(geometry, row="_fm_row_owner")
    matrix_coordinates = _peer_coordinates(geometry, row="_fm_row_owner[:, None]")
    source, ids, weight = (operands[name]["abi"] for name in ("dispatched", "router_expert_ids", "gate_weight"))
    lanes = int(source["scalar_lane_count"])
    k_coordinate = "_fm_k" if lanes == 1 else f"(_fm_k // {lanes})"
    input_offset = emit_local_scalar_offset(
        source,
        ("_fm_token", "_fm_local_route", k_coordinate),
        lane_coordinate=None if lanes == 1 else f"(_fm_k % {lanes})",
        shard_coordinates=route_coordinates,
    )
    local_coordinates = ("_fm_expert", "_fm_local_n[:, None]", "_fm_k[None, :]")
    logical_n = emit_logical_coordinate(
        weight,
        1,
        ("_fm_expert", "_fm_local_n", "_fm_k_start"),
        shard_coordinates=row_coordinates,
    )
    additions = {
        "chip_routes": geometry.chip_routes,
        "chip_rows": geometry.chip_rows,
        "local_routes": geometry.local_routes,
        "local_rows": geometry.local_rows,
        "row_owners": prod(geometry.hierarchy[i] for i in geometry.row_axes),
        "row_tiles": geometry.row_tiles,
        "k_blocks": geometry.k_blocks,
        "k_block_lanes": 1 << (geometry.k_blocks - 1).bit_length(),
        "raw_tasks": geometry.tokens * geometry.chip_routes * geometry.row_tiles * geometry.k_blocks,
        "route_owner_index": _linear(geometry, geometry.route_axes),
        "row_owner_index": _linear(geometry, geometry.row_axes),
        "peer_input": _peer_pointer(operands["dispatched"], route_coordinates),
        "peer_input_offset": input_offset,
        "peer_expert_ids": _peer_pointer(operands["router_expert_ids"], route_coordinates),
        "peer_expert_offset": emit_local_scalar_offset(
            ids,
            ("_fm_token", "_fm_local_route"),
            shard_coordinates=route_coordinates,
        ),
        "peer_slot_active": (
            f"(_fm_token < ({emit_active_extent(ids, 0, shard_coordinates=route_coordinates)})) & "
            f"(_fm_local_route < ({emit_active_extent(ids, 1, shard_coordinates=route_coordinates)}))"
        ),
        "peer_n_active": emit_active_extent(weight, 1, shard_coordinates=row_coordinates),
    }
    for field in ("gate", "up"):
        binding = operands[field + "_weight"]
        additions[field + "_peer_pointer"] = _peer_pointer(binding, matrix_coordinates)
        additions[field + "_peer_offset"] = emit_local_scalar_offset(
            binding["abi"],
            local_coordinates,
            shard_coordinates=matrix_coordinates,
        )
        scale = operands[field + "_proj_scale"]
        additions[field + "_peer_scale"] = _pointer(scale)
        additions[field + "_peer_scale_offset"] = emit_local_scalar_offset(
            scale["abi"],
            ("_fm_expert", f"(({logical_n}) // 128)", "_fm_k_block"),
        )
    return {**context, **additions}
