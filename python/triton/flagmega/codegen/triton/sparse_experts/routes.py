# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Standalone owner-local dispatch and route reduction."""

from triton.flagmega.codegen.triton.physical_access import emit_active_extent, emit_local_scalar_offset, emit_triton_scalar_type
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.model import Node, type_from_data
from triton.flagmega.ir.memory_effect import MemoryEffect
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.types import data_type, VectorType


def sparse_experts_routes_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _parameter, _scalar_local_domain, _canonical_writer_active

    definition = get_definition(raw["semantic_op"])
    operands = {p.name: _buffer(raw, "inputs", p.name) for p in definition.input_parameters if p.memory_effect != MemoryEffect.NONE}
    inputs = tuple(Node(p.name, "builtin.var", (), _tensor_type(operands[p.name]["abi"]) if p.name in operands
                        else type_from_data(_parameter(raw, "inputs", p.name)["type"])) for p in definition.input_parameters)
    result = _buffer(raw, "outputs", "result")
    if definition.prepare(inputs, raw["semantic_attrs"]).result_type != _tensor_type(result["abi"]):
        raise CodegenError("Expert route stage ABI disagrees with its owner contract")
    output = result["abi"]
    domain = _scalar_local_domain(output, "_fm_offsets")
    coords = domain["local_coordinates"]
    common = {
        "pointers": {name: _pointer(binding) for name, binding in operands.items()},
        "result": _pointer(result), "capacity": domain["capacity"], "active": domain["active"],
        "writer_active": _canonical_writer_active(output), "tile": 128,
        "result_offset": emit_local_scalar_offset(output, coords, lane_coordinate=domain["lane_coordinate"]),
    }
    if raw["semantic_op"] == "nn.sparse_experts_dispatch":
        return {**common, "source_offset": emit_local_scalar_offset(operands["value"]["abi"],
                                                                   (coords[0], coords[2]), lane_coordinate=domain["lane_coordinate"])}
    source = operands["projections"]["abi"]
    scalar = f"(({coords[-1]}) * {output['scalar_lane_count']} + ({domain['lane_coordinate'] or '0'}))"
    dtype = raw["semantic_attrs"]["output_dtype"]
    dtype = data_type(dtype) if dtype is not None else data_type(source["scalar_dtype"])
    dtype = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    return {
        **common,
        "routes": int(source["local_capacity_shape"][1]),
        "route_active": f"(_fm_route < ({emit_active_extent(source, 1)}))",
        "source_offset": last_axis_offset(source, (coords[0], "_fm_route"), scalar),
        "coefficient_offset": emit_local_scalar_offset(operands["router_expert_weights"]["abi"], (coords[0], "_fm_route")),
        "round_weighted": raw["semantic_attrs"]["round_weighted_output"],
        "weighted_dtype": emit_triton_scalar_type(dtype.value),
    }
