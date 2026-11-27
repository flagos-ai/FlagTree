# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bounded private-Shared geometry for ordered packed BF16 MMA."""

import re

from triton.flagmega.errors import CodegenError


def mma_prefetch_context(parameters, call):
    """Retain the logical address expressions with distinct bulk tensor names."""
    if call["block_k"] != 32 or call["tile_n"] != 16:
        raise CodegenError("MMA prefetch requires K32/N16 Tensor Core tiles.")
    capacity = call["local_k_capacity"]
    columns = call["local_n_capacity"]
    if capacity <= 0 or columns <= 0:
        raise CodegenError("MMA prefetch requires positive local K/N capacities.")
    budget = parameters.get("prefetch_elements")
    maximum = parameters.get("prefetch_max_k")
    for name, value in (("prefetch_elements", budget), ("prefetch_max_k", maximum)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0 or value & (value - 1):
            raise CodegenError(f"MMA {name} must be a positive power of two.")
    rows = 1 << (min(columns, call["tile_n"]) - 1).bit_length()
    if maximum < call["block_k"] or budget < rows * call["block_k"]:
        raise CodegenError("MMA prefetch budget cannot hold a complete K32 tile.")
    width = min(maximum, budget // rows, max(call["block_k"], 1 << (capacity - 1).bit_length()))
    result = {"prefetch_n": rows, "prefetch_k": width, "prefetch_shared_bytes": 2 * width * (rows + 1)}
    names = {
        "dense_local_n_offsets": "dense_bulk_n_offsets",
        "dense_global_n": "dense_bulk_global_n",
        "dense_local_k_offsets": "dense_bulk_k_offsets",
        "dense_global_k": "dense_bulk_global_k",
    }
    pattern = re.compile(r"\b(?:" + "|".join(names) + r")\b")
    for key in ("global_n", "global_k", "result_active", "source_active", "source_offset", "weight_offset"):
        result["prefetch_" + key] = pattern.sub(lambda match: names[match[0]], call[key])
    return result
