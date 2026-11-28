# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Collect explicit final-axis Pack demands without guessing vector widths."""

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
        if node.op != "tensors.pack" or node.inputs[0] not in source_ids:
            continue
        rank = tensor_of(module.node_map[node.inputs[0]].type).rank
        lanes = tuple(node.attrs["lanes"])
        axes = normalize_axes(tuple(node.attrs.get("axes", (node.attrs.get("axis", -1), ) * len(lanes))), rank)
        if axes and all(axis == rank - 1 for axis in axes):
            demands[lanes] += 1
    return min(demands, key=lambda lanes: (-demands[lanes], lanes)) if demands else None


__all__ = ["final_axis_pack_demand"]
