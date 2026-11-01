# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""TP4 row projection keeps one complete rounded K partition on each chip."""

from triton.flagmega import ir as fm


def owns_one_partition(kind):
    return (isinstance(kind, fm.DistributedType) and kind.partial == fm.SBPPartial((0,))
            and kind.exclusive is None and all(policy == fm.SBP.broadcast() for policy in kind.axis_policies))


def owns_material_partition(kind):
    return (isinstance(kind, fm.DistributedType) and kind.partial is None and kind.exclusive is None
            and kind.axis_policies == (fm.SBP.split_contiguous((0,)), fm.SBP.broadcast(), fm.SBP.broadcast()))


def selections(graph):
    selected = {}
    for node in graph.module.nodes:
        if node.op not in {"nn.partitioned_linear", "nn.partitioned_block_scaled_linear"}:
            continue
        if node.op == "nn.partitioned_linear" and node.attrs["partitions"] == 1:
            from .mla_input_projection import select_broadcast
            selected[node.id] = select_broadcast(graph,node).id
            continue
        predicate = owns_material_partition if node.op == "nn.partitioned_block_scaled_linear" else owns_one_partition
        candidates = [candidate for candidate in graph.bucket_map[node.id].candidates
                      if predicate(candidate.return_type)]
        if len(candidates) != 1:
            raise ValueError(f"{node.id}: expected one complete TP4 K partition candidate, got "
                             f"{[candidate.id for candidate in candidates]}")
        selected[node.id] = candidates[0].id
    return selected
