# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Named expert operands and scalar/vector local coordinate domains."""

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_local_scalar_offset,
    emit_triton_scalar_type,
)
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node


def last_axis_offset(abi, prefix, scalar):
    lane_count = int(abi["scalar_lane_count"])
    coordinate = scalar if lane_count == 1 else f"(({scalar}) // {lane_count})"
    lane = None if lane_count == 1 else f"(({scalar}) % {lane_count})"
    return emit_local_scalar_offset(abi, (*prefix, coordinate), lane_coordinate=lane)


def stage_context(raw, definition, weight_parameter):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _canonical_writer_active

    operands = {parameter.name: _buffer(raw, "inputs", parameter.name) for parameter in definition.input_parameters}
    inputs = tuple(
        Node(parameter.name, "builtin.var", (), _tensor_type(operands[parameter.name]["abi"]))
        for parameter in definition.input_parameters)
    result = _buffer(raw, "outputs", "result")
    attrs = raw["semantic_attrs"]
    if definition.prepare(inputs, attrs).result_type != _tensor_type(result["abi"]):
        raise CodegenError("Sparse expert result ABI disagrees with its typed sharding and rounding contract.")
    activation = operands[definition.input_parameters[0].name]["abi"]
    weight = operands[weight_parameter]["abi"]
    output = result["abi"]
    scalar_n = int(output["local_capacity_shape"][-1]) * int(output["scalar_lane_count"])
    scalar_k = int(activation["local_capacity_shape"][-1]) * int(activation["scalar_lane_count"])
    if tuple(weight["local_capacity_shape"][-2:]) != (scalar_n, scalar_k):
        raise CodegenError("Sparse expert local scalar N/K extents disagree with the weight ABI.")
    block_n, block_k = (int(raw["parameters"][name]) for name in ("block_n", "block_k"))
    if any(value <= 0 or value & (value - 1) for value in (block_n, block_k)):
        raise CodegenError("Sparse expert tiles must be positive powers of two.")
    context = {
        "pointers": {name: _pointer(binding)
                     for name, binding in operands.items()},
        "result": _pointer(result),
        "attrs": attrs,
        "output_dtype": emit_triton_scalar_type(output["scalar_dtype"]),
        "block_n": block_n,
        "block_k": block_k,
        "tokens": int(output["local_capacity_shape"][0]),
        "routes": int(operands["router_expert_ids"]["abi"]["local_capacity_shape"][1]),
        "scalar_n": scalar_n,
        "scalar_k": scalar_k,
        "token_active": f"(_fm_token < ({emit_active_extent(output, 0)}))",
        "n_active":
        f"(_fm_n < (({emit_active_extent(output, len(output['local_capacity_shape']) - 1)}) * {output['scalar_lane_count']}))",
        "k_active":
        f"(_fm_k < (({emit_active_extent(activation, len(activation['local_capacity_shape']) - 1)}) * {activation['scalar_lane_count']}))",
        "writer_active": _canonical_writer_active(output),
        "expert_offset": emit_local_scalar_offset(operands["router_expert_ids"]["abi"], ("_fm_token", "_fm_route")),
        "scale_offsets": {
            name: emit_local_scalar_offset(binding["abi"], ("_fm_expert", "0"))
            for name, binding in operands.items()
            if name.endswith("_scale")
        },
    }
    context["scale_loads"] = {
        name: (f"tl.full((), {binding['float32_splat']!r}, tl.float32)" if "float32_splat" in binding else
               f"tl.load({_pointer(binding)} + ({context['scale_offsets'][name]}), "
               f"mask={context['token_active']}, other=1)")
        for name, binding in operands.items()
        if name.endswith("_scale")
    }
    # div_rn is lowered as an explicit correctly-rounded operation, so even
    # a literal one can retain expensive device division. This identity is
    # exact (including signed zero); other constants keep the original divide.
    context["unit_scales"] = {
        name: binding.get("float32_splat") == 1.0
        for name, binding in operands.items()
        if name.endswith("_scale")
    }
    return context, operands, result
