# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered full-K projections with disjoint chip-shared output tasks."""

from triton.flagmega.codegen.triton.cooperative_access import require_chip_shared
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.sparse_experts.cooperative_gate_up import (
    _peer_coordinates,
    _peer_pointer,
    _workspace,
    gate_up_peer_context,
)
from triton.flagmega.codegen.triton.sparse_experts.cooperative_geometry import cooperative_gate_geometry
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError


def complete_k_gate_up_context(raw, context, operands, result):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _exclusive_participation_active, _pointer

    geometry = cooperative_gate_geometry(
        {name: _tensor_type(binding["abi"]) for name, binding in operands.items()}, _tensor_type(result["abi"]))
    facts = raw.get("facts", {})
    if (geometry is None or facts.get("internal_grid_barriers") != 1
            or facts.get("source_memory_scope") != "chip" or facts.get("result_memory_scope") != "chip"
            or _exclusive_participation_active(raw) != "True"
            or {v["formal"] for v in raw.get("workspaces", ())} != {"quantized_activations", "activation_scales"}):
        raise CodegenError("Complete-K experts require complete chip domains, chip effects and quantization scratch.")
    task_n = raw["parameters"].get("task_n")
    if type(task_n) is not int or task_n not in (32, 64):
        raise CodegenError("Complete-K experts require N32/N64 tasks with physical M64 WGMMA.")
    stages = raw["parameters"].get("num_stages", 1)
    if type(stages) is not int or stages not in (1, 2, 3, 4):
        raise CodegenError("Complete-K experts require one to four pipeline stages.")
    for name, binding in (*operands.items(), ("result", result)):
        require_chip_shared(binding, name, owner="Complete-K experts")
    quantized = _buffer(raw, "workspaces", "quantized_activations")
    scales = _buffer(raw, "workspaces", "activation_scales")
    require_chip_shared(quantized, "quantized_activations", owner="Complete-K experts")
    quantized_shape = (geometry.tokens, geometry.chip_routes, geometry.reduction)
    qabi = quantized["abi"]
    if (qabi.get("coordinate_space") != "canonical_global" or qabi.get("scalar_dtype") != "float8_e4m3fn"
            or qabi.get("scalar_lane_count") != 1 or tuple(qabi["local_capacity_shape"]) != quantized_shape
            or tuple(qabi["scalar_storage_strides"]) != (geometry.chip_routes * geometry.reduction, geometry.reduction, 1)):
        raise CodegenError("Complete-K experts require dense FP8 quantized activation scratch.")
    _workspace(scales, geometry.scale_shape, "activation_scales")
    coordinates = _peer_coordinates(geometry, route="_fm_route_owner", row="_fm_row_owner")
    lanes = int(result["abi"]["scalar_lane_count"])
    result_offset = emit_local_scalar_offset(result["abi"],
        ("_fm_token", "_fm_local_route", f"(_fm_local_n // {lanes})"),
        lane_coordinate=f"(_fm_local_n % {lanes})" if lanes != 1 else None, shard_coordinates=coordinates)
    tiles = (geometry.chip_rows + task_n - 1) // task_n
    return {
        **gate_up_peer_context(geometry, context, operands),
        "internal_grid_barriers": 1,
        "quantized_pointer": _pointer(quantized), "quantization_scale_pointer": _pointer(scales),
        "quantization_tasks": geometry.tokens * geometry.chip_routes * geometry.k_blocks,
        "scale_loads": {name: expression.replace(context["slot_active"], "_fm_active_slot")
                        for name, expression in context["scale_loads"].items()},
        "task_n": task_n, "matrix_rows": 2 * task_n, "task_tiles": tiles, "num_stages": stages,
        "complete_tasks": geometry.tokens * geometry.chip_routes * tiles,
        "result_peer_pointer": _peer_pointer(result, coordinates), "result_peer_offset": result_offset,
    }
