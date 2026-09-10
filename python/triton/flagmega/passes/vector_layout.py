# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Coordinate AutoVectorize defaults across shape-preserving dataflow regions."""

from collections import Counter, defaultdict
from copy import copy
from dataclasses import replace
from math import prod

from triton.flagmega.ir import TensorType


def region_candidates(rule, node, module, lane_bytes, lanes=None):
    """Regenerate and validate explicit lane alternatives, also during resume."""
    if not getattr(rule, "layout_input_indices", ()) or not isinstance(lane_bytes, int) or lane_bytes <= 0:
        return ()
    configured = copy(rule)
    configured.lane_bytes = lane_bytes
    value = module.node_map[node.inputs[rule.layout_input_indices[0]]]
    results = []
    for candidate in configured.candidates(node, module):
        if candidate.axes != (value.type.rank - 1, ) or len(candidate.lanes) != 1:
            continue
        layout = tuple(lanes) if lanes is not None else candidate.lanes
        if prod(layout) != prod(candidate.lanes):
            continue
        axes = (value.type.rank - 1, ) * len(layout)
        parameters = {**candidate.parameters, "region_lane_bytes": lane_bytes, "region_lanes": layout}
        for name, content in (("axes", axes), ("value_axes", axes), ("lanes", layout)):
            if name in parameters:
                parameters[name] = content
        if "weight_axes" in parameters:
            parameters["weight_axes"] = (0, ) * len(layout)
        results.append(
            replace(candidate, id=f"{candidate.id}.lanes_{'_'.join(map(str, layout))}", axes=axes, lanes=layout,
                    parameters=parameters))
    return tuple(results)


def coordinate_layouts(module, registry, alternatives, choose_default):
    """Offer common producer lanes without changing any user-selected layout.

    Union only relationships declared by eligible rules and proven shape-equal.
    Call relations follow the declared formal/result ABI, including tuple
    projections. Views, collectives and state are not guessed through.
    Among conflicting anchors prefer the most shared width, then smaller lanes;
    this is a deterministic boundary-count heuristic, not a hardware cost model.
    """
    rules = {rule.name: rule for rule in registry.rules}
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(lhs, rhs):
        if isinstance(lhs.type, TensorType) and lhs.type == rhs.type:
            parent[find(lhs.id)] = find(rhs.id)

    # A reused callee has one ABI for all invocations. Coordinate its callers
    # before independent consumer choices create competing boundary demands.
    for node in module.nodes:
        if node.op == "builtin.call":
            function = module.function_map.get(str(node.attrs["callee"]))
            if function is None:
                continue
            for actual, formal in zip(node.inputs, function.parameters):
                join(module.node_map[actual], module.node_map[formal])
            if len(function.outputs) == 1:
                join(node, module.node_map[function.outputs[0]])
        elif node.op == "builtin.get_item" and len(node.inputs) == 1:
            call = module.node_map[node.inputs[0]]
            if call.op != "builtin.call":
                continue
            function = module.function_map.get(str(call.attrs["callee"]))
            index = int(node.attrs["index"])
            if function is not None and 0 <= index < len(function.outputs):
                join(node, module.node_map[function.outputs[index]])

    owners = {}
    defaults = {}
    for node in module.nodes:
        candidates = alternatives.get(node.id, ())
        if not candidates:
            continue
        default = choose_default(candidates)
        defaults[node.id] = default
        rule = rules[next(value.rule for value in candidates if value.id == default)]
        indices = getattr(rule, "layout_input_indices", ())
        if not indices:
            continue
        values = [module.node_map[node.inputs[index]] for index in indices]
        if getattr(rule, "layout_output", True):
            values.append(node)
        if any(not isinstance(value.type, TensorType) or value.type.shape != values[0].type.shape for value in values):
            continue
        for value in values[1:]:
            parent[find(value.id)] = find(values[0].id)
        owners[node.id] = (rule, values[0])

    anchors = defaultdict(Counter)
    for node in module.nodes:
        candidates = alternatives.get(node.id, ())
        if not candidates:
            continue
        candidate = next(value for value in candidates if value.id == defaults[node.id])
        if (getattr(rules[candidate.rule], "output_layout_anchor", False) and isinstance(node.type, TensorType)
                and candidate.axes and all(axis == node.type.rank - 1 for axis in candidate.axes)):
            anchors[find(node.id)][candidate.lanes] += 1

    enriched = dict(alternatives)
    for node_id, (rule, value) in owners.items():
        demands = anchors.get(find(value.id))
        if not demands:
            continue
        lanes = min(demands, key=lambda lane: (-demands[lane], lane))
        lane_bytes = prod(lanes) * value.type.dtype.itemsize
        if lane_bytes == rule.lane_bytes and len(lanes) == 1:
            continue
        candidates = region_candidates(rule, module.node_map[node_id], module, lane_bytes, lanes)
        if not candidates:
            continue
        enriched[node_id] = (*alternatives[node_id], *candidates)
        defaults[node_id] = candidates[0].id
    return enriched, defaults
