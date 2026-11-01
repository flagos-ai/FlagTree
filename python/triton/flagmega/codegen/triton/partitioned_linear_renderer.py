# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical dense matrix proof for complete and chip-partitioned K domains."""

from triton.flagmega.codegen.triton.cooperative_access import require_chip_shared
from triton.flagmega.codegen.triton.descriptor_abi import device_descriptor_request
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DistributedType, SBPSplit


def partitioned_linear_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _static_shape

    bindings = {name: _buffer(raw, section, name) for section, name in
                (("inputs", "value"), ("inputs", "weight"), ("outputs", "result"))}
    shapes, strides, origins, physical_shapes = {}, {}, {}, {}
    for name, binding in bindings.items():
        require_chip_shared(binding, name, owner="PartitionedLinear")
        abi = binding["abi"]
        if abi["scalar_dtype"] != "bfloat16":
            raise CodegenError("PartitionedLinear requires BF16 physical buffers.")
        shape = _static_shape(abi, "local_capacity_shape")
        lanes = int(abi.get("scalar_lane_count", 1))
        shapes[name] = (shape[0], shape[1] * lanes)
        strides[name] = _static_shape(abi, "scalar_storage_strides")
        logical = _static_shape(abi, "logical_shape") if abi.get("coordinate_space") == "canonical_global" else shape
        physical_shapes[name] = (logical[0], logical[1] * lanes)
        if strides[name] != (physical_shapes[name][1], lanes):
            raise CodegenError("PartitionedLinear requires row-major buffers with only final-axis lanes.")
        origins[name] = emit_local_scalar_offset(abi, ("0", "0"))
    x, w, out = (bindings[name] for name in ("value", "weight", "result"))
    if out["abi"].get("storage_kind") == "compact_per_owner":
        raise CodegenError("PartitionedLinear cooperates on one chip result; materialize public partials through Boxing.")
    reduction, rows = shapes["weight"][1], shapes["weight"][0]
    if shapes["value"] != (1, reduction) or shapes["result"] != (1, rows):
        raise CodegenError("PartitionedLinear physical input, weight and result extents disagree.")
    kind = _tensor_type(w["abi"])
    parts = 1 if isinstance(kind, DistributedType) and isinstance(kind.axis_policies[1], SBPSplit) else raw["semantic_attrs"]["partitions"]
    span, block_k = reduction // parts, int(raw["parameters"]["block_k"])
    splits = int(raw["semantic_attrs"].get("accumulation_splits", 1))
    if splits <= 0 or span % splits:
        raise CodegenError("PartitionedLinear requires complete FP32 accumulation splits.")
    tile_n = int(raw['parameters'].get('tile_n', 64))
    recipe = None
    if raw['variant'] == 'simt':
        if tile_n <= 0 or tile_n & (tile_n - 1) or block_k <= 0 or block_k & (block_k - 1):
            raise CodegenError('PartitionedLinear SIMT tiles must be positive powers of two.')
        mapping = raw['parameters'].get('task_mapping')
        if mapping is not None:
            from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
            recipe = TaskLaunchRecipe.from_data(mapping)
            if recipe.work_size != rows or recipe.tile_size != tile_n:
                raise CodegenError('PartitionedLinear task domain must cover exactly its local scalar output rows.')
    context = {
        "rows": rows, "reduction": reduction, "parts": parts, "part_k": span,
        "tasks": (rows + tile_n - 1) // tile_n, 'tile_n': tile_n, "block_k": block_k,
        "accumulation_splits": splits, "accumulation_k": span // splits,
        "source": f"({_pointer(x)} + ({origins['value']}))",
        "weight": f"({_pointer(w)} + ({origins['weight']}))",
        "result": f"({_pointer(out)} + ({origins['result']}))",
        "weight_stride": strides["weight"][0],
        **({'task_launch_recipe': recipe.to_data()} if recipe is not None else {}),
    }
    if raw["variant"] != "wgmma_local":
        return context
    if (span // splits) % block_k:
        raise CodegenError("PartitionedLinear TMA must not read across a semantic partition or FP32 sub-dot boundary.")
    workspaces = tuple(raw["shared_workspaces"])
    if len(workspaces) != 2:
        raise CodegenError("PartitionedLinear TMA requires two typed Shared workspaces.")
    roots = {}
    for value in workspaces:
        roots.setdefault(value["physical_buffer"], []).append(value)
    offsets, cursor = {}, 0
    for values in roots.values():
        alignment = max(value["alignment_bytes"] for value in values)
        cursor = (cursor + alignment - 1) // alignment * alignment
        base = min(value["offset_bytes"] for value in values)
        for value in values:
            offsets[value["name"]] = cursor + value["offset_bytes"] - base
        cursor += max(value["offset_bytes"] + value["nbytes"] for value in values) - base
    intervals = sorted((offsets[value["name"]], offsets[value["name"]] + value["nbytes"]) for value in workspaces)
    if intervals[0][1] > intervals[1][0]:
        raise CodegenError("PartitionedLinear simultaneously live Shared buffers overlap.")
    requests = []
    for name, binding, tile in (("value", x, 8), ("weight", w, 64)):
        shape = physical_shapes[name]
        if max(shape) >= 2**31:
            raise CodegenError("PartitionedLinear TMA coordinates must fit signed int32.")
        requests.append(device_descriptor_request({
            "parameter": f"{name}_descriptor", "source": str(binding["runtime_argument"]),
            "kind": "single", "offset_bytes": int(binding["abi"].get("pool_byte_offset", 0)),
            "dtype": "bfloat16", "shape": shape, "strides": (shape[1], 1),
            "block_shape": (tile, block_k), "source_shape_axes": ((), ()), "padding": "zero",
        }, workspaces))
    return {**context, "local_shared_bytes": cursor,
            "local_shared_alignment": max(value["alignment_bytes"] for value in workspaces),
            "local_weight_offset": offsets["weight_tiles"], "local_activation_offset": offsets["activation_tiles"],
            "source_k_origin": origins["value"],
            "weight_row_origin": f"(({origins['weight']}) // {physical_shapes['weight'][1]})",
            "weight_k_origin": f"(({origins['weight']}) % {physical_shapes['weight'][1]})",
            "host_tensor_descriptor_requests": tuple(requests)}


__all__ = ["partitioned_linear_call"]
