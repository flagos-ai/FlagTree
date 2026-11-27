# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FP32 sum over local axes, retaining distributed partial components."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_scalar_immediate
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.codegen.triton.fusion import decode_fusion_attrs, emit_body, raw_semantics, triton_type
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import Node
from triton.flagmega.ir.axis import normalize_axis
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.math.reduce_sum import ReduceSum
from triton.flagmega.ir.types import VectorType
from triton.flagmega.ir.op_fusion import has_ops


def reduce_sum_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _canonical_writer_active, _pointer

    source = _buffer(raw, "inputs", "value")
    result = _buffer(raw, "outputs", "result")
    input_type = _tensor_type(source["abi"])
    attrs = decode_fusion_attrs(raw["semantic_attrs"])
    if ReduceSum.infer_call_type((Node("source", "builtin.var", (), input_type), ), attrs) != _tensor_type(result["abi"]):
        raise CodegenError("ReduceSum result ABI disagrees with its local/partial reduction contract.")
    rank = len(source["abi"]["local_capacity_shape"])
    axes = tuple(normalize_axis(axis, rank) for axis in attrs["axes"])
    domain = local_reduction_domain(source["abi"], axes, int(raw["parameters"]["elements_per_program"]), lane_wise=True)
    tiled = {}
    if raw["variant"] == "tiled_rows":
        row_tile = int(raw["parameters"]["row_tile"])
        if row_tile <= 0 or row_tile & (row_tile - 1):
            raise CodegenError("A reduction row tile must be a positive power of two.")
        # Spend the existing element budget on independent rows when K is short.
        # Large K retains the original one-row reduction and register footprint.
        row_tile = min(row_tile, 1 << (max(domain["rows"], 1) - 1).bit_length(),
                       int(raw["parameters"]["elements_per_program"]) // domain["tile"])
        tiled_domain = local_reduction_domain(
            source["abi"], axes, int(raw["parameters"]["elements_per_program"]),
            row_index="_fm_row[:, None]", reduction_index="_fm_offsets[None, :]", lane_wise=True)
        tiled = {"row_tile": row_tile, "tiled_active": tiled_domain["active"],
                 "tiled_source_offset": emit_local_scalar_offset(source["abi"], tiled_domain["coordinates"],
                                                                  lane_coordinate=tiled_domain.get("lane_coordinate"))}
    output_coordinates = (tuple("0" if axis in axes else coordinate
                                for axis, coordinate in enumerate(domain["coordinates"]))
                          if attrs["keep_dims"] else domain["outer_coordinates"])
    output_type = tensor_of(_tensor_type(result["abi"]))
    lane_coordinate = domain.get("lane_coordinate")
    if isinstance(output_type.dtype, VectorType) and lane_coordinate is not None and output_coordinates:
        lanes = output_type.dtype.lane_count
        source_lanes = int(source["abi"].get("scalar_lane_count", 1))
        scalar_column = f"(({output_coordinates[-1]}) * {source_lanes} + ({lane_coordinate}))"
        output_coordinates = (*output_coordinates[:-1], f"(({scalar_column}) // {lanes})")
        lane_coordinate = f"(({scalar_column}) % {lanes})"
    elif isinstance(output_type.dtype, VectorType) and lane_coordinate is None:
        lanes = output_type.dtype.lane_count
        scalar_column = output_coordinates[-1]
        output_coordinates = (*output_coordinates[:-1], f"(({scalar_column}) // {lanes})")
        if lanes > 1:
            lane_coordinate = f"(({scalar_column}) % {lanes})"
    scalar = source["abi"]["storage"] == "scalar"
    scalar_value = None
    if scalar:
        scalar_value = (emit_scalar_immediate(source["abi"], source["runtime_argument"])
                        if source["runtime_value_kind"] == "immediate" else source["runtime_argument"])
    program = {}
    if has_ops(attrs):
        _, base, _ = raw_semantics(raw)
        body = attrs.get("pre_ops", {}).get("value")
        pre, value = emit_body(body, "_fm_values", "_fm_sum_pre")
        identity, identity_value = emit_body(body, "_fm_accumulator", "_fm_sum_identity_pre")
        post, post_value = emit_body(next(iter(attrs.get("post_ops", ())), None),
                                    f"(_fm_accumulator).to({triton_type(base.type)})", "_fm_sum_post")
        program = {"fusion_pre_lines": pre, "fusion_pre_value": value,
                   "fusion_identity_lines": identity, "fusion_identity_value": identity_value,
                   "fusion_post_lines": post, "fusion_post_value": post_value}
    return {
        **program,
        **domain,
        **tiled,
        "identity": not axes,
        "scalar_value": scalar_value,
        "source": None if scalar else _pointer(source),
        "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"],
                                                   lane_coordinate=domain.get("lane_coordinate")),
        "result": _pointer(result),
        "result_offset": emit_local_scalar_offset(result["abi"], output_coordinates, lane_coordinate=lane_coordinate),
        "writer_active": _canonical_writer_active(result["abi"]),
    }
