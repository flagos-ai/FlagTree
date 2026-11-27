# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Chip-peer FP8 projection addresses and explicit ordered-merge workspaces."""

from math import prod

from triton.flagmega.codegen.triton.block_fp8_geometry import cooperative_block_geometry
from triton.flagmega.codegen.triton.cooperative_access import (
    chip_peer_pointer,
    require_chip_shared,
    require_chip_workspace,
    require_owner_input,
    require_owner_output,
)
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_global_scalar_offset,
    emit_local_scalar_offset,
)
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError


def cooperative_block_context(raw, context, operands, result, *, partial_name="projection_partials",
                              owner="Cooperative FP8 projections"):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _canonical_writer_active,
        _exclusive_participation_active,
        _pointer,
    )

    geometry = cooperative_block_geometry(
        {name: _tensor_type(binding["abi"]) for name, binding in operands.items()},
        _tensor_type(result["abi"]),
        raw["semantic_attrs"],
    )
    if geometry is None:
        raise CodegenError(f"{owner} require a complete contiguous chip owner domain.")
    if (
        raw.get("facts", {}).get("internal_grid_barriers") != 1
        or raw.get("facts", {}).get("source_memory_scope") != "chip"
        or _exclusive_participation_active(raw) != "True"
        or context["scaled_accumulation"] != "fma"
        or context["tile_n"] != 16
        or context["block_k"] != 128
    ):
        raise CodegenError(f"{owner} require all CTAs, chip read effects, one barrier and ordered N16/K128 FMA.")
    partials = _buffer(raw, "workspaces", partial_name)
    scales = _buffer(raw, "workspaces", "activation_scales")
    for name, binding in operands.items():
        if name == "value" and raw.get("facts", {}).get("source_memory_scopes", {}).get(name) in {"block", "inferred"}:
            require_owner_input(binding, name, owner=owner)
        else:
            require_chip_shared(binding, name, owner=owner)
    require_owner_output(result, "result", owner=owner)
    require_chip_workspace(partials, geometry.partial_shape, partial_name, owner=owner)
    require_chip_workspace(scales, geometry.scale_shape, "activation_scales", owner=owner)

    coordinates = [f"shard_coord{i}" if level != "b" else "0" for i, level in enumerate(geometry.levels)]
    linear = "0"
    for position, axis in enumerate(geometry.row_axes):
        stride = prod(geometry.hierarchy[i] for i in geometry.row_axes[position + 1 :])
        coordinates[axis] = f"(((_fm_owner[:, None]) // {stride}) % {geometry.hierarchy[axis]})"
        linear = f"({linear} * {geometry.hierarchy[axis]} + shard_coord{axis})"
    weight = operands["weight"]
    lanes = int(weight["abi"]["scalar_lane_count"])
    k_coordinate = "_fm_k[None, :]" if lanes == 1 else f"(_fm_k[None, :] // {lanes})"
    scale = operands["weight_scale"]
    return {
        **context,
        "internal_grid_barriers": 1,
        "projection_pointer": _pointer(partials),
        "activation_scale_pointer": _pointer(scales),
        "chip_rows": geometry.chip_rows,
        "local_rows": geometry.local_rows,
        "row_tiles": geometry.row_tiles,
        "row_owners": prod(geometry.hierarchy[i] for i in geometry.row_axes),
        "row_owner_index": linear,
        "k_blocks": geometry.k_blocks,
        "k_block_lanes": 1 << (geometry.k_blocks - 1).bit_length(),
        "raw_tasks": geometry.row_tiles * geometry.k_blocks,
        "source_pointer": _pointer(operands["value"]),
        "source_offset": emit_local_scalar_offset(operands["value"]["abi"], ("0", "_fm_k")),
        "weight_peer_pointer": chip_peer_pointer(weight, coordinates, owner=owner),
        "weight_peer_offset": emit_local_scalar_offset(
            weight["abi"],
            ("_fm_local_n[:, None]", k_coordinate),
            lane_coordinate=None if lanes == 1 else f"(_fm_k[None, :] % {lanes})",
            shard_coordinates=coordinates,
        ),
        "peer_rows_active": emit_active_extent(
            weight["abi"],
            0,
            shard_coordinates=tuple(value.replace("[:, None]", "") for value in coordinates),
        ),
        "merge_scale_offset": emit_global_scalar_offset(
            scale["abi"],
            (f"(_fm_global_n[None, :] // {raw['semantic_attrs']['weight_block_n']})", "_fm_merge_k[:, None]"),
        ),
        "merge_writer": _canonical_writer_active(result["abi"]),
    }
