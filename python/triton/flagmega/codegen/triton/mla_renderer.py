# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Address a head's K/V projection rows directly through the verified ABI."""

from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.model import DistributedType, NoneType, SBP, TensorType
from triton.flagmega.ir.distributed_type import SBPSplit
from triton.flagmega.ir.ops.nn._mla_head_distribution import validate_mla_head_distribution
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent, emit_local_scalar_offset, emit_logical_coordinate,
)


def mla_projection_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer,
        _canonical_writer_active,
        _pointer,
        _static_shape,
    )

    attrs = raw["semantic_attrs"]
    x, w, result = (
        _buffer(raw, group, name) for group, name in (("inputs", "value"), ("inputs", "weight"), ("outputs", "result"))
    )
    shape = _static_shape(x["abi"], "logical_shape")
    if len(shape) != 3 or shape[0] != 1:
        raise CodegenError("MLA projection MMA currently implements single-token rank-3 inputs.")
    if x["abi"]["scalar_dtype"] != "bfloat16" or result["abi"]["scalar_dtype"] != "bfloat16":
        raise CodegenError("MLA projection MMA requires BF16 values/results.")
    weight_dtype = str(w["abi"]["scalar_dtype"])
    if weight_dtype not in {"bfloat16", "float8_e4m3fn"}:
        raise CodegenError("MLA projection MMA requires BF16 or FP8 E4M3 weights.")
    quantized = weight_dtype == "float8_e4m3fn"
    scale = _buffer(raw, "inputs", "scale") if quantized else None
    bindings = (x, w, *((scale,) if scale else ()), result)
    placement = next((kind.placement for binding in bindings
                      if isinstance(kind := _tensor_type(binding["abi"]), DistributedType)), None)

    def ownership_type(binding):
        abi = binding["abi"]
        kind = _tensor_type(abi)
        if placement is not None and isinstance(kind, TensorType):
            # Canonical arguments/results may retain a plain physical tensor
            # ABI beside a distributed private activation. A complete such
            # tensor is a broadcast value for ownership proofs; its storage
            # and addressing stay unchanged. This cannot create head shards.
            if (abi.get("coordinate_space") != "canonical_global"
                    or abi.get("distributed_backing_type") is not None
                    or tuple(abi["local_capacity_shape"]) != tuple(abi["logical_shape"])):
                raise CodegenError("MLA projection requires a complete canonical tensor for implicit broadcast ownership.")
            return DistributedType(kind, (SBP.broadcast(),) * kind.rank, placement)
        return kind

    source_type = ownership_type(x)
    result_type = ownership_type(result)
    try:
        validate_mla_head_distribution(source_type, ownership_type(w),
                                       ownership_type(scale) if scale else NoneType(), attrs)
    except IRSchemaError as error:
        raise CodegenError(str(error)) from error
    if (isinstance(source_type, DistributedType) != isinstance(result_type, DistributedType)
            or isinstance(source_type, DistributedType) and (
                source_type.placement != result_type.placement
                or source_type.axis_policies != result_type.axis_policies
                or result_type.partial is not None or result_type.exclusive is not None)):
        raise CodegenError("MLA projection result must preserve source head ownership.")
    head_sharded = isinstance(source_type, DistributedType) and isinstance(source_type.axis_policies[1], SBPSplit)
    owned = {}
    if head_sharded:
        head = "_fm_mla_head"
        reduction_index = "_fm_mla_reduction"
        columns = "_fm_mla_columns"
        def packed_offset(binding, scalar):
            lanes = int(binding["abi"].get("scalar_lane_count", 1))
            return emit_local_scalar_offset(binding["abi"], ("0", head, f"({scalar} // {lanes})"),
                                            lane_coordinate=f"({scalar} % {lanes})" if lanes > 1 else None)
        owned["source_offset"] = packed_offset(x, reduction_index)
        owned["result_offset"] = packed_offset(result, columns)
        owned["global_head"] = emit_logical_coordinate(x["abi"], 1, ("0", head, "0"))
        owned["head_active"] = f"({head} < ({emit_active_extent(x['abi'], 1)}))"
        owned["weight_offset"] = emit_local_scalar_offset(w["abi"], ("_fm_mla_weight_row", "_fm_mla_weight_col"))
        if scale:
            scale_type = _tensor_type(scale["abi"])
            scale_sharded = isinstance(scale_type, DistributedType) and isinstance(scale_type.axis_policies[0], SBPSplit)
            scale_row = "_fm_mla_weight_row" if scale_sharded else "_fm_mla_global_weight_row"
            owned["scale_offset"] = emit_local_scalar_offset(scale["abi"],
                (f"({scale_row} // {attrs['weight_block_n']})", f"(_fm_mla_weight_col // {attrs['weight_block_k']})"))
    x_strides = x["abi"]["scalar_storage_strides"]
    w_strides = w["abi"]["scalar_storage_strides"]
    result_strides = result["abi"]["scalar_storage_strides"]
    reduction = attrs["qk_head_dim"] if attrs["direction"] == "absorb_query" else attrs["latent_dim"]
    output = attrs["latent_dim"] if attrs["direction"] == "absorb_query" else attrs["value_head_dim"]
    parameters = raw.get("parameters", {})
    return {
        "value": _pointer(x),
        "weight": _pointer(w),
        "scale": _pointer(scale) if scale else None,
        "result": _pointer(result),
        "writer_active": _canonical_writer_active(result["abi"]),
        "direction": attrs["direction"],
        "heads": _static_shape(x["abi"], "local_capacity_shape")[1] if head_sharded else shape[1],
        "head_sharded": head_sharded,
        **owned,
        "reduction": reduction,
        "output": output,
        "qk_dim": attrs["qk_head_dim"],
        "value_dim": attrs["value_head_dim"],
        "source_head_stride": int(x_strides[-2]),
        "source_dim_stride": int(x_strides[-1]),
        "source_lanes": int(x["abi"].get("scalar_lane_count", 1)),
        "result_head_stride": int(result_strides[-2]),
        "result_dim_stride": int(result_strides[-1]),
        "result_lanes": int(result["abi"].get("scalar_lane_count", 1)),
        "weight_row_stride": int(w_strides[0]),
        "weight_col_stride": int(w_strides[1]),
        "quantized": quantized,
        "block_n": int(attrs["weight_block_n"]),
        "block_k": int(attrs["weight_block_k"]),
        "scale_row_stride": int(scale["abi"]["scalar_storage_strides"][0]) if scale else 0,
        "scale_col_stride": int(scale["abi"]["scalar_storage_strides"][1]) if scale else 0,
        "tile_n": int(parameters.get("tile_n", 64)),
        "reduction_tile": int(parameters.get("block_k", 64)),
    }
