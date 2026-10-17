# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Address a head's K/V projection rows directly through the verified ABI."""

from collections.abc import Mapping

from triton.flagmega.errors import CodegenError


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
    for value in (x, w, result):
        distributed = value["abi"].get("distributed_type")
        if isinstance(distributed, Mapping) and (
            distributed.get("partial") is not None
            or distributed.get("exclusive") is not None
            or any(p.get("kind") == "split" for p in distributed.get("axis_policies", ()))
        ):
            raise CodegenError("MLA projection requires fully broadcast buffers.")
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
        "heads": shape[1],
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
