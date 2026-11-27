# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove dense chip views and bind six independent raw-weight tensor maps."""

from math import prod

from triton.flagmega.codegen.triton.descriptor_abi import device_descriptor_request
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.errors import CodegenError


def kda_input_projection_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _static_shape

    x = _buffer(raw, "inputs", "value")
    names = ("query_weight", "key_weight", "value_weight", "beta_weight", "forget_weight", "output_weight")
    weights = tuple(_buffer(raw, "inputs", name) for name in names)
    outputs = tuple(_buffer(raw, "outputs", f"result_{i}") for i in range(6))
    tile = int(raw["parameters"]["tile_n"])
    reduction_tile = int(raw["parameters"]["block_k"])
    tma = raw["variant"] in {"wgmma", "wgmma_local"}
    if tile != 64:
        raise CodegenError("KDA input projection implements 64-row tiles.")

    def origin(binding):
        return emit_local_scalar_offset(binding["abi"], ("0",) * len(binding["abi"]["logical_shape"]))

    def dense(binding):
        abi = binding["abi"]
        if abi["scalar_dtype"] != "bfloat16":
            raise CodegenError("KDA input projection ABI requires BF16 buffers.")
        shape = _static_shape(abi, "local_capacity_shape")
        strides = _static_shape(abi, "scalar_storage_strides")
        span = int(abi.get("scalar_lane_count", 1))
        for extent, stride in reversed(tuple(zip(shape, strides, strict=True))):
            if extent > 1 and stride != span:
                raise CodegenError("KDA input projection requires dense head-major chip buffers.")
            span *= extent
        return shape, int(abi.get("scalar_lane_count", 1))

    shape, lanes = dense(x)
    if shape[0] != 1:
        raise CodegenError("KDA input projection requires one token.")
    reduction = shape[-1] * lanes
    fields, requests, start = [], [{
        "parameter": "source_descriptor", "source": str(x["runtime_argument"]), "kind": "single",
        "offset_bytes": int(x["abi"].get("pool_byte_offset", 0)), "dtype": "bfloat16",
        "shape": (1, reduction), "strides": (reduction, 1), "block_shape": (8, reduction_tile),
        "source_shape_axes": ((), ()), "padding": "zero",
    }], 0
    for i, (w, out) in enumerate(zip(weights, outputs, strict=True)):
        w_shape, w_lanes = dense(w)
        out_shape, out_lanes = dense(out)
        if w_lanes != 1 or w_shape[-1] != reduction:
            raise CodegenError("KDA input projection requires complete K and scalar raw weights.")
        rows = prod(w_shape[:-1])
        if prod(out_shape) * out_lanes != rows:
            raise CodegenError("KDA input projection output capacity disagrees with its weight rows.")
        tasks = (rows + tile - 1) // tile
        fields.append({"index": i, "rows": rows, "start": start, "end": start + tasks,
                       "weight": f"({_pointer(w)} + ({origin(w)}))",
                       "output": f"({_pointer(out)} + ({origin(out)}))",
                       "row_origin": f"(({origin(w)}) // {reduction})"})
        start += tasks
        descriptor_shape = (_static_shape(w["abi"], "logical_shape")
                            if w["abi"].get("coordinate_space") == "canonical_global" else w_shape)
        if tma and max(prod(descriptor_shape[:-1]), reduction) >= 2**31:
            raise CodegenError("KDA TMA coordinates must fit signed int32.")
        requests.append({
            "parameter": f"weight_descriptor_{i}", "source": str(w["runtime_argument"]),
            "kind": "single", "offset_bytes": int(w["abi"].get("pool_byte_offset", 0)),
            "dtype": "bfloat16", "shape": (prod(descriptor_shape[:-1]), reduction),
            "strides": (reduction, 1), "block_shape": (tile, reduction_tile),
            "source_shape_axes": ((), ()), "padding": "zero",
        })
    for field in fields:
        field["branch"] = ("else:" if field["index"] == 5 else
                           f"{'if' if field['index'] == 0 else 'elif'} _fm_task < {field['end']}:")
    resources = {}
    if raw["variant"] == "wgmma_local":
        workspaces = tuple(raw["shared_workspaces"])
        if len(workspaces) != 2:
            raise CodegenError("Local KDA WGMMA requires two typed Shared views.")
        # Non-pipelined workspaces may have independent physical roots. Map
        # each root into this invocation's frame without aliasing distinct
        # allocations; preserve every within-root offset and alignment proof.
        groups = {}
        for workspace in workspaces:
            groups.setdefault(workspace["physical_buffer"], []).append(workspace)
        cursor, offsets = 0, {}
        for group in groups.values():
            alignment = max(w["alignment_bytes"] for w in group)
            cursor = (cursor + alignment - 1) // alignment * alignment
            base = min(w["offset_bytes"] for w in group)
            for w in group:
                offsets[w["name"]] = cursor + w["offset_bytes"] - base
            cursor += max(w["offset_bytes"] + w["nbytes"] for w in group) - base
        intervals = sorted((offsets[w["name"]], offsets[w["name"]] + w["nbytes"]) for w in workspaces)
        if intervals[0][1] > intervals[1][0]:
            raise CodegenError("Local KDA WGMMA weight and activation live simultaneously.")
        resources = {"local_shared_bytes": cursor,
                     "local_shared_alignment": max(w["alignment_bytes"] for w in workspaces),
                     "local_weight_offset": offsets["weight"],
                     "local_activation_offset": offsets["activation"]}
        requests = tuple(device_descriptor_request(request, workspaces) for request in requests)
    return {**resources, "source": f"({_pointer(x)} + ({origin(x)}))", "reduction": reduction,
            "block_k": reduction_tile, "tile_n": tile, "fields": tuple(fields), "tasks": start,
            "host_tensor_descriptor_requests": tuple(requests) if tma else ()}
