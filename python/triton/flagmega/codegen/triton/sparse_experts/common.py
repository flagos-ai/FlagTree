# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Named expert operands and scalar/vector local coordinate domains."""

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent,
    emit_local_scalar_offset,
    emit_logical_coordinate,
    emit_triton_scalar_type,
)
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, Node, local_shard_descriptor


def last_axis_offset(abi, prefix, scalar):
    lane_count = int(abi["scalar_lane_count"])
    coordinate = scalar if lane_count == 1 else f"(({scalar}) // {lane_count})"
    lane = None if lane_count == 1 else f"(({scalar}) % {lane_count})"
    return emit_local_scalar_offset(abi, (*prefix, coordinate), lane_coordinate=lane)


def _axis_capacity_is_active(value_type, axis, capacity):
    if isinstance(value_type, DistributedType):
        coordinates = tuple(f"owner_{index}" for index in range(value_type.placement.rank))
        extent = local_shard_descriptor(value_type, coordinates).active_shape[axis]
    else:
        extent = value_type.shape[axis]
    return extent.is_fixed and extent.fixed_value == capacity


def stage_context(raw, definition, weight_parameter, *, partitioned=False):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _canonical_writer_active, _pointer

    operands = {parameter.name: _buffer(raw, "inputs", parameter.name) for parameter in definition.input_parameters}
    inputs = tuple(
        Node(parameter.name, "builtin.var", (), _tensor_type(operands[parameter.name]["abi"]))
        for parameter in definition.input_parameters)
    result = _buffer(raw, "outputs", "result")
    attrs = raw["semantic_attrs"]
    if definition.prepare(inputs, attrs).result_type != _tensor_type(result["abi"]):
        raise CodegenError("Sparse expert result ABI disagrees with its typed sharding and rounding contract.")
    if raw.get("variant") == "wgmma":
        from triton.flagmega.codegen.triton.sparse_experts.mma import supports_block_fp8_wgmma
        if (not supports_block_fp8_wgmma(raw["semantic_op"], {
                parameter.name: value.type
                for parameter, value in zip(definition.input_parameters, inputs, strict=True)
        }, attrs) or raw["parameters"]["block_n"] != 64 or raw["parameters"]["block_k"] != 128):
            raise CodegenError("Expert WGMMA requires BF16/FP8, complete K, UE8M0 and 128x128 block scales.")
    activation = operands[definition.input_parameters[0].name]["abi"]
    weight = operands[weight_parameter]["abi"]
    output = result["abi"]
    token_axis = 1 if partitioned else 0
    weight_n_axis = 2 if partitioned else 1
    weight_k_axis = 3 if partitioned else 2
    scalar_n = int(output["local_capacity_shape"][-1]) * int(output["scalar_lane_count"])
    scalar_k = int(activation["local_capacity_shape"][-1]) * int(activation["scalar_lane_count"])
    if tuple(weight["local_capacity_shape"][-2:]) != (scalar_n, scalar_k):
        raise CodegenError("Sparse expert local scalar N/K extents disagree with the weight ABI.")
    block_n, block_k = (int(raw["parameters"][name]) for name in ("block_n", "block_k"))
    weight_block_n = int(attrs["weight_block_n"])
    weight_block_k = int(attrs["weight_block_k"])
    if any(value <= 0 or value & (value - 1) for value in (block_n, block_k, weight_block_n, weight_block_k)):
        raise CodegenError("Sparse expert tiles must be positive powers of two.")
    routes_abi = operands["router_expert_ids"]["abi"]
    routes = int(routes_abi["local_capacity_shape"][1])
    # The loop already bounds capacity. Repeating that bound in its loads
    # retains redundant predicates through software-pipeline prologues.
    full_routes = _axis_capacity_is_active(_tensor_type(routes_abi), 1, routes)
    # A weight block scale is a per-K-group contract, and the K tile is an
    # independent tuning parameter. One scalar group index per tile is only
    # equivalent while a tile stays inside a single group; once a tile spans
    # several groups every K lane needs its own group, both for the weight
    # scale and for the activation quantization the scale was computed against.
    groups_per_tile = block_k // weight_block_k if block_k > weight_block_k else 1
    scale_offsets = {}
    block_scale_loads = {}
    for name, binding in operands.items():
        if not name.endswith("_scale"):
            continue
        scale_rank = len(binding["abi"]["local_capacity_shape"])
        if scale_rank == 2:
            scale_offsets[name] = emit_local_scalar_offset(binding["abi"], ("_fm_expert", "0"))
        elif scale_rank == 3 or (partitioned and scale_rank == 4):
            # Weights are compact owner slices; their scales are broadcast
            # global tables. Recover the logical weight coordinates before
            # selecting a scale block, including split-N and split-K owners.
            coordinates = (("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")
                           if groups_per_tile > 1 else
                           ("_fm_expert", "_fm_n", "_fm_k_start"))
            if partitioned:
                coordinates = (coordinates[0], "_fm_partition", *coordinates[1:])
            logical_row = emit_logical_coordinate(weight, weight_n_axis, coordinates)
            logical_column = emit_logical_coordinate(weight, weight_k_axis, coordinates)
            row = f"({logical_row} // {weight_block_n})"
            column = f"({logical_column} // {weight_block_k})"
            scale_coordinates = ("_fm_expert", row, column)
            if partitioned:
                logical_partition = emit_logical_coordinate(weight, 1, coordinates)
                scale_coordinates = ("_fm_expert", logical_partition, row, column)
            scale_offset = emit_local_scalar_offset(binding["abi"], scale_coordinates)
            # Either form multiplies an (N, K) weight tile, so the per-tile
            # column keeps the trailing broadcast the per-lane form has already.
            block_scale_loads[name] = (f"tl.load({_pointer(binding)} + ({scale_offset}), mask={{group_mask}}, other=1)"
                                       if groups_per_tile > 1 else
                                       f"tl.load({_pointer(binding)} + ({scale_offset}), mask={{slot_active}}, other=1)[:, None]")
        else:
            raise CodegenError(f"Sparse expert scale {name!r} must have rank 2 or 3.")
    context = {
        "pointers": {name: _pointer(binding)
                     for name, binding in operands.items()},
        "result": _pointer(result),
        "attrs": attrs,
        "output_dtype": emit_triton_scalar_type(output["scalar_dtype"]),
        "activation_dtype": emit_triton_scalar_type(activation["scalar_dtype"]),
        "block_n": block_n,
        "block_k": block_k,
        "tokens": int(output["local_capacity_shape"][token_axis]),
        "routes": routes,
        "scalar_n": scalar_n,
        "scalar_k": scalar_k,
        "token_active": f"(_fm_token < ({emit_active_extent(output, token_axis)}))",
        "route_active": "True" if full_routes else f"(_fm_route < ({emit_active_extent(routes_abi, 1)}))",
        "n_active":
        f"(_fm_n < (({emit_active_extent(output, len(output['local_capacity_shape']) - 1)}) * {output['scalar_lane_count']}))",
        "k_active":
        f"(_fm_k < (({emit_active_extent(activation, len(activation['local_capacity_shape']) - 1)}) * {activation['scalar_lane_count']}))",
        "writer_active": _canonical_writer_active(output),
        "expert_offset": emit_local_scalar_offset(operands["router_expert_ids"]["abi"], ("_fm_token", "_fm_route")),
        "scale_offsets": scale_offsets,
        "block_scale_loads": block_scale_loads,
        "weight_block_k": weight_block_k,
        "groups_per_tile": groups_per_tile,
    }
    if partitioned:
        context["partitions"] = int(output["local_capacity_shape"][0])
        context["token_active"] += f" & (_fm_partition < ({emit_active_extent(output, 0)}))"
    context["slot_active"] = (context["token_active"] if full_routes else
                              f"({context['token_active']}) & ({context['route_active']})")
    context["scale_loads"] = {
        name: (f"tl.full((), {binding['float32_splat']!r}, tl.float32)" if "float32_splat" in binding else
               (f"tl.load({_pointer(binding)} + ({context['scale_offsets'][name]}), "
                f"mask={context['slot_active']}, other=1)" if name in context["scale_offsets"] else
                "tl.full((), 1.0, tl.float32)"))
        for name, binding in operands.items()
        if name.endswith("_scale")
    }
    # A per-lane group index reads one scale per (N, K) weight element, so its
    # predicate is the weight tile's, not the scalar slot's.
    group_mask = f"_fm_n_mask[:, None] & _fm_k_mask[None, :]" if groups_per_tile > 1 else context["slot_active"]
    context["block_scale_loads"] = {
        name: expression.replace("{slot_active}", "_fm_n_mask" if raw.get("variant") == "wgmma"
                                 else context["slot_active"]).replace("{group_mask}", group_mask)
        for name, expression in context["block_scale_loads"].items()
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
