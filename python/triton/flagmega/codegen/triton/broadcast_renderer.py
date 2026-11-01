# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit owner-local broadcasting, preserving outer/vector units."""

from math import prod
from triton.flagmega.codegen.triton.fusion import decode_fusion_attrs, emit_body, raw_semantics, triton_type
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_scalar_immediate
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.broadcast_to import BroadcastTo
from triton.flagmega.ir.op_fusion import has_ops


def broadcast_to_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _pointer,
        _scalar_local_domain,
        _canonical_writer_active,
    )

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    input_type, output_type = _tensor_type(source["abi"]), _tensor_type(result["abi"])
    attrs = decode_fusion_attrs(raw["semantic_attrs"])
    if BroadcastTo.infer_call_type((Node("source", "builtin.var", (), input_type), ), attrs) != output_type:
        raise CodegenError("BroadcastTo ABI does not match the operation's owner mapping.")
    input_tensor, output_tensor = tensor_of(input_type), tensor_of(output_type)
    domain = _scalar_local_domain(result["abi"], "_fm_offsets")
    offset = output_tensor.rank - input_tensor.rank
    coordinates = tuple("0" if dim.is_fixed and dim.fixed_value == 1 else domain["local_coordinates"][axis + offset]
                        for axis, dim in enumerate(input_tensor.shape))
    input_lanes = getattr(input_tensor.dtype, "lanes", ())
    output_lanes = getattr(output_tensor.dtype, "lanes", ())
    source_lane = None
    if input_lanes:
        lane_offset = len(output_lanes) - len(input_lanes)
        terms = [
            f"((({domain['lane_coordinate']}) // {prod(output_lanes[i + lane_offset + 1:])}) % {extent})"
            f" * {prod(input_lanes[i + 1:])}" for i, extent in enumerate(input_lanes) if extent != 1
        ]
        source_lane = " + ".join(terms) or "0"
    scalar = source["abi"]["storage"] == "scalar"
    source_value = None
    if scalar:
        source_value = (emit_scalar_immediate(source["abi"], source["runtime_argument"])
                        if source["runtime_value_kind"] == "immediate" else source["runtime_argument"])
    program = {}
    if has_ops(attrs):
        _, base, _ = raw_semantics(raw)
        pre, value = emit_body(attrs.get("pre_ops", {}).get("value"), "_fm_value", "_fm_broadcast_pre")
        post, value = emit_body(next(iter(attrs.get("post_ops", ())), None),
                               f"({value}).to({triton_type(base.type)})", "_fm_broadcast_post")
        program = {"fusion_lines": (*pre, *post), "fusion_result": value}
    return {
        **program,
        "writer_active":
        _canonical_writer_active(result["abi"]),
        "source":
        None if scalar else _pointer(source),
        "scalar_value":
        source_value,
        "source_offset":
        None if scalar else emit_local_scalar_offset(source["abi"], coordinates, lane_coordinate=source_lane),
        "result":
        _pointer(result),
        "result_offset":
        emit_local_scalar_offset(result["abi"], domain["local_coordinates"], lane_coordinate=domain["lane_coordinate"]),
        "capacity":
        domain["capacity"],
        "active":
        domain["active"],
        "tile":
        int(raw["parameters"]["elements_per_program"]),
    }
