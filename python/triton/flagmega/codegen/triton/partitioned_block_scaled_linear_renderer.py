# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical local partitions with globally indexed broadcast quantization scales."""

from triton.flagmega.codegen.triton.cooperative_access import require_chip_shared
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_logical_coordinate
from triton.flagmega.errors import CodegenError


def partitioned_block_scaled_linear_call(raw):
    from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer, _pointer, _static_shape

    bindings = {name: _buffer(raw, section, name) for section, name in
                (("inputs", "value"), ("inputs", "weight"), ("inputs", "weight_scale"), ("outputs", "result"))}
    for name, binding in bindings.items():
        require_chip_shared(binding, name, owner="PartitionedBlockScaledLinear")
    x, w, scale, out = (bindings[name] for name in ("value", "weight", "weight_scale", "result"))
    if any(binding["abi"]["scalar_dtype"] != dtype for binding, dtype in
           ((x, "bfloat16"), (w, "float8_e4m3fn"), (scale, "float32"), (out, "bfloat16"))):
        raise CodegenError("PartitionedBlockScaledLinear physical dtypes must match its BF16/FP8/FP32 contract.")
    shapes = {name: _static_shape(binding["abi"], "local_capacity_shape") for name, binding in bindings.items()}
    input_width, output_width = (int(binding["abi"].get("scalar_lane_count", 1)) for binding in (x, out))
    parts, tokens, row_groups = shapes["result"]
    rows, reduction = shapes["weight"]
    attrs = raw["semantic_attrs"]
    total_k = _static_shape(w["abi"], "logical_shape")[1]
    span = total_k // attrs["partitions"]
    if (shapes["value"] != (tokens, reduction // input_width) or reduction != parts * span
            or rows != row_groups * output_width
            or int(w["abi"].get("scalar_lane_count", 1)) != 1
            or int(scale["abi"].get("scalar_lane_count", 1)) != 1
            or attrs["weight_block_k"] != 128 or span % 128):
        raise CodegenError("PartitionedBlockScaledLinear physical owners must contain complete local P/T/N/K domains.")
    tile = int(raw["parameters"]["tile_n"])
    accumulator_layout = raw["parameters"].get("accumulator_layout", "vector")
    if accumulator_layout not in {"vector", "mma"}:
        raise CodegenError("PartitionedBlockScaledLinear accumulator layout must be vector or mma.")
    activation_layout = raw["parameters"].get("activation_layout", "auto")
    if activation_layout not in {"auto", "warp"}:
        raise CodegenError("PartitionedBlockScaledLinear activation layout must be auto or warp.")
    warps = int(raw["parameters"]["compute_num_warps"])
    if warps < 4 or warps % 4:
        raise CodegenError("PartitionedBlockScaledLinear native FP8 MMA requires a multiple of four warps.")
    mma_columns = int(raw["parameters"].get("mma_columns", 16))
    # The current MMA-v3 lowering requires N % 16 == 0. N8 selects MMA-v2,
    # whose Hopper FP8 path promotes operands to FP16 and changes arithmetic.
    if mma_columns != 16:
        raise CodegenError("PartitionedBlockScaledLinear native FP8 MMA requires 16 columns on the current lowering.")
    if tile not in {16, 32, 64} or int(raw["parameters"]["block_k"]) != 128:
        raise CodegenError("PartitionedBlockScaledLinear requires N16/32/64 tiles and K128 blocks.")
    groups = raw['parameters'].get('k_blocks_per_mma', 1)
    if type(groups) is not int or groups not in (1, 2, 4, 8, 16):
        raise CodegenError('PartitionedBlockScaledLinear K groups must be 1/2/4/8/16.')
    if groups > 1 and accumulator_layout != 'vector':
        raise CodegenError('Independent K groups require an ordered vector accumulator.')
    row_order = raw['parameters'].get('k_group_row_order', 'group_major')
    if row_order not in ('group_major', 'n_major'):
        raise CodegenError('PartitionedBlockScaledLinear K group row order must be group_major or n_major.')
    tasks = parts * tokens * ((rows + tile - 1) // tile)
    mapping = raw['parameters'].get('task_mapping')
    recipe = None
    if mapping is not None:
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
        recipe = TaskLaunchRecipe.from_data(mapping)
        if recipe.work_size != tasks or recipe.tile_size != 1:
            raise CodegenError('PartitionedBlockScaledLinear task domain must cover exactly its complete P/T/N tiles.')
    local_k = f"(_fm_part * {span} + {'_fm_source_k' if groups > 1 else '_fm_k'})"
    weight_k = f"(_fm_part * {span} + _fm_weight_k)" if groups > 1 else local_k.replace("_fm_k", "_fm_k[None, :]")
    global_p = emit_logical_coordinate(out["abi"], 0, ("_fm_part", "_fm_token", f"(_fm_n // {output_width})"))
    global_n = emit_logical_coordinate(out["abi"], 2, ("_fm_part", "_fm_token", f"(_fm_n // {output_width})"))
    global_n = f"(({global_n}) * {output_width} + (_fm_n % {output_width}))"
    return {
        # Task ownership and the matrix instruction shape are independent.
        # M<64 selects FP16 MMA after FP8 conversion, changing the native FP8
        # dot's arithmetic. Pad inactive rows while retaining smaller tasks.
        "parts": parts, "tokens": tokens, "rows": rows, "part_k": span,
        "task_tile_n": tile, "tile_n": max(tile * groups, 64),
        "k_blocks_per_mma": groups, "dot_k": 128,
        "k_group_row_order": row_order,
        "accumulator_layout": accumulator_layout,
        "activation_layout": activation_layout,
        "mma_columns": mma_columns,
        "row_tiles": (rows + tile - 1) // tile,
        "tasks": tasks,
        **({'task_launch_recipe': recipe.to_data()} if recipe is not None else {}),
        "activation_quantization": attrs["activation_quantization"],
        "source": _pointer(x), "weight": _pointer(w), "scale": _pointer(scale), "result": _pointer(out),
        "source_offset": emit_local_scalar_offset(x["abi"], ("_fm_token", f"({local_k} // {input_width})"),
            lane_coordinate=f"({local_k} % {input_width})" if input_width != 1 else None),
        "weight_offset": emit_local_scalar_offset(w["abi"], ("_fm_n[:, None]", weight_k)),
        "scale_offset": emit_local_scalar_offset(scale["abi"],
            (f"(({global_n}) // {attrs['weight_block_n']})", f"(({global_p}) * {span // 128} + _fm_block)")),
        "result_offset": emit_local_scalar_offset(out["abi"], ("_fm_part", "_fm_token", f"(_fm_n // {output_width})"),
            lane_coordinate=f"(_fm_n % {output_width})" if output_width != 1 else None),
    }


__all__ = ["partitioned_block_scaled_linear_call"]
