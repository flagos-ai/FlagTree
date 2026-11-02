# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Collect final-axis Pack demands (explicit or folded into an operand) without guessing vector widths."""

from collections import Counter

from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.tensors.pack import normalize_axes


def final_axis_pack_demand(module, source_ids):
    """Select the most shared exact layout; other users retain their ABI.

    Stable tie-breaking is independent of SSA names, models and targets.
    Callers must validate that the producer can implement this layout.
    """
    demands = Counter()
    for node in module.nodes:
        if node.op == "tensors.pack":
            packs = ((node.inputs[0], node.attrs), )
        elif node.op == "math.vectorized_binary":
            # A folded operand pack is the same demand as an explicit Pack of it.
            packs = tuple((value, node.attrs[name]) for value, name in zip(node.inputs, ("lhs_pack", "rhs_pack"))
                          if node.attrs.get(name) is not None and "padding" not in node.attrs[name])
        else:
            continue
        for value, packing in packs:
            if value not in source_ids:
                continue
            rank = tensor_of(module.node_map[value].type).rank
            lanes = tuple(packing["lanes"])
            axes = normalize_axes(tuple(packing.get("axes", (packing.get("axis", -1), ) * len(lanes))), rank)
            if axes and all(axis == rank - 1 for axis in axes):
                demands[lanes] += 1
    return min(demands, key=lambda lanes: (-demands[lanes], lanes)) if demands else None


__all__ = ["final_axis_pack_demand"]
