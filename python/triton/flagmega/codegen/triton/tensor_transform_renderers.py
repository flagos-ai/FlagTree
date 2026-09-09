# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Pad/Slice address transforms over local tensor shards, including vector lanes.

As in nncase's NTT Pad/Slice, these kernels transform local coordinates only.
Axes whose index range changes must be broadcast; any inter-owner data
movement is represented by an explicit Boxing in the input graph.
"""

from math import prod

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, Node, SBP, get_definition, tensor_type, type_from_data, vector_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.slice import Slice
from triton.flagmega.ir.ops.tensors.pack import normalize_axes


def tensor_transform_call(raw):
    # The shared call-binding/domain helpers do not depend on an op's indexing
    # semantics. Import lazily because the family registry owns this encoder.
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _scalar_local_domain

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    input_abi, output_abi = source["abi"], result["abi"]
    source_type = _tensor_type(input_abi)
    result_type = _tensor_type(output_abi)
    input_tensor, output_tensor = tensor_of(source_type), tensor_of(result_type)
    if input_tensor.dtype != output_tensor.dtype or input_tensor.rank != output_tensor.rank:
        raise CodegenError("Pad/Slice require matching scalar/vector dtype and rank.")
    domain = _scalar_local_domain(output_abi, "offsets")
    coordinates = list(domain["local_coordinates"])
    attrs = raw["semantic_attrs"]
    starts, steps = [0] * input_tensor.rank, [1] * input_tensor.rank
    if raw["semantic_op"] == "tensors.slice":
        for axis, indices in Slice.ranges(input_tensor, attrs):
            starts[axis], steps[axis] = indices.start, indices.step
    masks = []
    for axis, (start, step) in enumerate(zip(starts, steps)):
        changed = input_tensor.shape[axis] != output_tensor.shape[axis] or start != 0 or step != 1
        if isinstance(source_type, DistributedType) or isinstance(result_type, DistributedType):
            if not isinstance(source_type, DistributedType) or not isinstance(result_type, DistributedType):
                raise CodegenError("Pad/Slice require one explicit distribution on both operands.")
            if (source_type.placement != result_type.placement or source_type.partial != result_type.partial
                    or source_type.axis_policies[axis] != result_type.axis_policies[axis]):
                raise CodegenError("Pad/Slice owner mappings differ; insert explicit Boxing.")
            if changed and source_type.axis_policies[axis] != SBP.broadcast():
                raise CodegenError("Pad/Slice changed axes must be broadcast; insert explicit Boxing.")
        if changed:
            coordinate = f"({start} + ({coordinates[axis]}) * {step})"
            bound = input_tensor.shape[axis].fixed_value
            mask = f"(({coordinate}) >= 0) & (({coordinate}) < {bound})"
            masks.append(f"({mask})")
            coordinates[axis] = f"tl.where({mask}, {coordinate}, 0)"
    return {
        "source":
        _pointer(source),
        "result":
        _pointer(result),
        "capacity":
        domain["capacity"],
        "active":
        domain["active"],
        "input_active":
        " & ".join(masks) or "True",
        "source_offset":
        emit_local_scalar_offset(input_abi, coordinates, lane_coordinate=domain["lane_coordinate"]),
        "result_offset":
        emit_local_scalar_offset(output_abi, domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
        "pad_value":
        attrs.get("pad_value", 0),
        "tile":
        int(raw["parameters"]["elements_per_program"]),
    }


def _tensor_type(abi):
    distributed = abi.get("distributed_type")
    if distributed is not None:
        return type_from_data(distributed)
    dtype = abi["scalar_dtype"]
    if abi["scalar_lane_shape"]:
        dtype = vector_type(dtype, abi["scalar_lane_shape"])
    return tensor_type(dtype, abi["logical_shape"])


def vector_relayout_call(raw):
    """Realize non-byte-preserving Pack/Unpack as explicit lane permutation.

    Zero-copy cases are removed by TIR lowering before candidate dispatch.
    Repeated axes and existing vector lanes follow Pack/Unpack's type contract,
    including scaled split units; the kernel needs no particular mesh shape.
    """
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _scalar_local_domain

    source, result = _buffer(raw, "inputs", "value"), _buffer(raw, "outputs", "result")
    input_abi, output_abi = source["abi"], result["abi"]
    source_type, result_type = _tensor_type(input_abi), _tensor_type(output_abi)
    definition = get_definition(raw["semantic_op"])
    attrs = definition.normalize_attrs(raw["semantic_attrs"])
    actual_type = definition.infer_type((Node("source", "builtin.var", (), source_type), ), attrs)
    if actual_type != result_type:
        raise CodegenError("Pack/Unpack call ABI disagrees with its inferred vector/layout contract.")
    packing = raw["semantic_op"] == "tensors.pack"
    input_lanes = tuple(input_abi["scalar_lane_shape"])
    output_lanes = tuple(output_abi["scalar_lane_shape"])
    lane_count = len(attrs["lanes"]) if packing else len(input_lanes)
    axes = normalize_axes(attrs.get("axes", (attrs.get("axis", -1), ) * lane_count), tensor_of(source_type).rank)
    lanes = tuple(attrs["lanes"]) if packing else input_lanes[:len(axes)]
    domain = _scalar_local_domain(output_abi, "offsets")
    coordinates = list(domain["local_coordinates"])
    output_lane = domain["lane_coordinate"] or "0"
    if packing:
        for index, (axis, lane) in enumerate(zip(axes, lanes)):
            stride = prod(output_lanes[index + 1:])
            digit = f"((({output_lane}) // {stride}) % {lane})"
            coordinates[axis] = f"(({coordinates[axis]}) * {lane} + {digit})"
        source_lane = f"(({output_lane}) % {prod(input_lanes)})" if input_lanes else None
    else:
        digits = [""] * len(lanes)
        for index in reversed(range(len(lanes))):
            axis, lane = axes[index], lanes[index]
            digits[index] = f"(({coordinates[axis]}) % {lane})"
            coordinates[axis] = f"(({coordinates[axis]}) // {lane})"
        source_lane = "0"
        for digit, lane in zip(digits, lanes):
            source_lane = f"(({source_lane}) * {lane} + ({digit}))"
        source_lane = f"(({source_lane}) * {prod(output_lanes)} + ({output_lane}))"
    return {
        "source":
        _pointer(source),
        "result":
        _pointer(result),
        "capacity":
        domain["capacity"],
        "active":
        domain["active"],
        "input_active":
        "True",
        "source_offset":
        emit_local_scalar_offset(input_abi, coordinates, lane_coordinate=source_lane),
        "result_offset":
        emit_local_scalar_offset(output_abi, domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
        "pad_value":
        0,
        "tile":
        int(raw["parameters"]["elements_per_program"]),
    }
