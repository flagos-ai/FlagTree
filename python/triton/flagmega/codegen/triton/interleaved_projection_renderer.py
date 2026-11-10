# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Map original native rows directly into two independently allocated fields."""

from math import prod

from triton.flagmega.codegen.triton.physical_access import (
    emit_active_extent, emit_local_scalar_offset,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, type_from_data
from triton.flagmega.ir.distributed_type import SBPBroadCast, SBPSplit, scale_split_units


def interleaved_projection_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import (
        _buffer, _pointer, _static_shape,
    )

    bindings = {name: _buffer(raw, section, name) for section, name in (
        ("inputs", "value"), ("inputs", "weight"), ("inputs", "weight_scale"),
        ("outputs", "result_0"), ("outputs", "result_1"),
    )}
    x, w, s, first, second = (bindings[n] for n in bindings)
    attrs = raw["semantic_attrs"]
    encoding = attrs.get("weight_encoding")
    if encoding not in {"IQ4_XS", "Q4_K"}:
        raise CodegenError("InterleavedProjection requires native IQ4_XS or Q4_K rows.")
    try:
        heads, width = tuple(attrs["feature_shape"])
        declared_lanes = tuple(tuple(lanes) for lanes in attrs.get("output_lanes", ((), ())))
    except (KeyError, TypeError, ValueError) as error:
        raise CodegenError("InterleavedProjection requires two fixed head/width fields.") from error
    if (type(heads) is not int or type(width) is not int or heads <= 0 or width <= 0
            or attrs.get("field_count", 2) != 2 or len(declared_lanes) != 2
            or any(any(type(lane) is not int or lane <= 0 for lane in lanes) for lanes in declared_lanes)):
        raise CodegenError("InterleavedProjection requires positive head/width and two lane shapes.")
    lane_counts = tuple(prod(lanes) for lanes in declared_lanes)
    if any(width % lanes for lanes in lane_counts):
        raise CodegenError("InterleavedProjection output lanes must divide the scalar field width.")
    shape = {n: _static_shape(b["abi"], "logical_shape") for n, b in bindings.items()}
    local = {n: _static_shape(b["abi"], "local_capacity_shape") for n, b in bindings.items()}
    if len(shape["value"]) != 2:
        raise CodegenError("InterleavedProjection requires rank-two scalar activations.")
    tokens, k = shape["value"]
    payload = (136 if encoding == "IQ4_XS" else 144) * (k // 256)
    if (k <= 0 or k % 256 or shape["weight"] != (heads * 2 * width, payload)
            or local["value"] != (tokens, k) or local["weight"][1:] != (payload,)):
        raise CodegenError("InterleavedProjection requires complete K256 blocks and original whole native rows.")
    if (x["abi"].get("scalar_dtype") not in {"bfloat16", "float32"}
            or w["abi"].get("scalar_dtype") != "uint8"
            or s["abi"].get("scalar_dtype") != "float32"
            or any(tuple(b["abi"].get("scalar_lane_shape", ())) for b in (x, w, s))
            or any(int(b["abi"].get("scalar_lane_count", 1)) != 1 for b in (x, w, s))
            or any(tuple(b["abi"]["scalar_storage_strides"])[-1] != 1 for b in (x, w))):
        raise CodegenError("InterleavedProjection requires scalar BF16/F32 activation, UINT8 bytes and FP32 scales with contiguous K.")
    bn, bk = attrs.get("weight_block_n"), attrs.get("weight_block_k")
    if (type(bn) is not int or type(bk) is not int or bn <= 0 or bk <= 0
            or shape["weight_scale"] != ((heads * 2 * width + bn - 1) // bn, (k + bk - 1) // bk)):
        raise CodegenError("InterleavedProjection scales must retain the logical N/K placeholder shape.")
    for index, result in enumerate((first, second)):
        name = f"result_{index}"
        if (shape[name] != (tokens, heads, width // lane_counts[index])
                or result["abi"].get("scalar_dtype") != "bfloat16"
                or tuple(result["abi"].get("scalar_lane_shape", ())) != declared_lanes[index]
                or int(result["abi"].get("scalar_lane_count", 1)) != lane_counts[index]
                or len(local[name]) != 3 or local[name][0] != tokens
                or local[name][2] != width // lane_counts[index]):
            raise CodegenError("InterleavedProjection physical results must match both declared BF16 fields.")
    if local["result_0"][:2] != local["result_1"][:2]:
        raise CodegenError("InterleavedProjection result fields must share complete token/head owners.")

    types = [type_from_data(b["abi"]["distributed_type"])
             if b["abi"].get("distributed_type") is not None else None for b in (x, w, s, first, second)]
    if any(t is not None for t in types):
        if not all(isinstance(t, DistributedType) for t in types):
            raise CodegenError("InterleavedProjection operands and fields must share distributed ownership.")
        xt, wt, st, a, b = types
        if (len({t.placement for t in types}) != 1 or any(t.partial is not None or t.exclusive is not None for t in types)
                or any(not isinstance(policy, SBPBroadCast) for t in (xt, st) for policy in t.axis_policies)
                or not isinstance(wt.axis_policies[1], SBPBroadCast)
                or a.axis_policies != b.axis_policies
                or any(not isinstance(a.axis_policies[axis], SBPBroadCast) for axis in (0, 2))):
            raise CodegenError("InterleavedProjection requires broadcast activation/K and identical whole-head result owners.")
        rows = wt.axis_policies[0]
        head = scale_split_units(rows, 1, 2 * width) if isinstance(rows, SBPSplit) else rows
        if not isinstance(rows, (SBPBroadCast, SBPSplit)) or head != a.axis_policies[1]:
            raise CodegenError("InterleavedProjection weight row owners must map exactly to whole result heads.")
    if local["weight"][0] != local["result_0"][1] * 2 * width:
        raise CodegenError("InterleavedProjection local native rows must cover both complete fields per head.")
    tile = raw["parameters"].get("tile_n")
    if type(tile) is not int or tile <= 0 or tile & (tile - 1):
        raise CodegenError("InterleavedProjection tile_n must be a positive power of two.")
    local_head = f"(interleaved_offsets // {2 * width})"
    scalar_d = f"(interleaved_offsets % {width})"
    field = f"((interleaved_offsets // {width}) % 2)"
    coords = tuple(("interleaved_m", local_head, f"({scalar_d} // {lanes})") for lanes in lane_counts)
    # The row/head split-unit proof above makes the source row local to this
    # owner. Let its ABI map that row into canonical or compact storage;
    # inverting a global coordinate would obscure this proof and introduce
    # unnecessary floor divisions into the C++ expression contract.
    local_row = f"(({local_head} * 2 + {field}) * {width} + {scalar_d})"
    mask = f"(interleaved_offsets < {local['result_0'][1] * 2 * width}) & ({local_head} < ({emit_active_extent(first['abi'], 1)})) & (interleaved_m < ({emit_active_extent(first['abi'], 0)}))"
    return {
        "value": _pointer(x), "weight": _pointer(w), "result_0": _pointer(first), "result_1": _pointer(second),
        "source_row_offset": emit_local_scalar_offset(x["abi"], ("interleaved_m", "0")),
        "weight_row_offset": emit_local_scalar_offset(w["abi"], (local_row, "0")),
        "destination_0_offset": emit_local_scalar_offset(first["abi"], coords[0], lane_coordinate=f"({scalar_d} % {lane_counts[0]})" if lane_counts[0] > 1 else None),
        "destination_1_offset": emit_local_scalar_offset(second["abi"], coords[1], lane_coordinate=f"({scalar_d} % {lane_counts[1]})" if lane_counts[1] > 1 else None),
        "field": field, "n_mask": mask, "local_m": tokens,
        "local_n_capacity": local["result_0"][1] * 2 * width,
        "k": k, "weight_encoding": encoding, "tile_n": tile,
    }
