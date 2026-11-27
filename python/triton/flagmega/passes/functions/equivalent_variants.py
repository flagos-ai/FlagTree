# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Retire function specializations whose finalized compilation contracts agree."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace

from triton.flagmega.ir import Function, IRModule, verify_module
from triton.flagmega.passes.functions.graph import function_nodes


# These fields record the transformations that led to the current signature.
# The exact signature, body, effects and selected implementation are compared
# separately. Unknown attributes/metadata always participate in equivalence.
_FUNCTION_HISTORY = frozenset({
    "specialized_from",
    "norm_stats_boundary_specialization",
    "post_auto_distributed_boundary_layout",
})
_NODE_HISTORY = frozenset({"introduced_by", "cloned_for_function_variant", "norm_stats_boundary"})


def merge_equivalent_function_variants(module: IRModule) -> IRModule:
    """Merge alpha-equivalent compiler-created graph variants into their source.

    Run after boundary layout planning reaches its fixed point, before TIR
    selection/bufferization. In particular, a partial-stats signature may have
    become materialized again. Keeping that obsolete specialization duplicates
    both its lowering work and its device function. Public functions without
    ``specialized_from`` are never coalesced.
    """

    if module.prim_functions or module.kernel_definitions or module.execution_functions:
        raise ValueError("Function variant merging must precede TIR materialization.")
    functions = module.function_map
    variants = tuple(f for f in module.functions if f.name != module.entry and f.attrs.get("specialized_from") in functions)
    if not variants:
        return module
    bodies = {f.name: {n.id for n in function_nodes(module, f)} for f in module.functions}
    owners: dict[str, set[str]] = defaultdict(set)
    for name, body in bodies.items():
        for node_id in body:
            owners[node_id].add(name)
    points = defaultdict(list)
    for point in module.selection_points:
        points[point.owner].append(point)
    records = module.selection_map
    replacements: dict[str, str] = {}
    removed_nodes: set[str] = set()
    for variant in variants:
        source = functions[str(variant.attrs["specialized_from"])]
        # Avoid cycles in user-edited specialization provenance.
        ancestors = {variant.name}
        current = source
        while current.name not in ancestors:
            ancestors.add(current.name)
            parent = functions.get(str(current.attrs.get("specialized_from", "")))
            if parent is None:
                break
            current = parent
        else:
            continue
        mapping = _equivalent_nodes(module, source, variant, owners, points, records)
        if mapping is None:
            continue
        replacements[variant.name] = source.name
        removed_nodes.update(node_id for node_id in bodies[variant.name] if owners[node_id] == {variant.name})
    if not replacements:
        return module

    def canonical(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    nodes = tuple(
        replace(node, attrs={**dict(node.attrs), "callee": canonical(str(node.attrs["callee"]))})
        if node.op == "builtin.call" and node.attrs.get("callee") in replacements else node
        for node in module.nodes if node.id not in removed_nodes
    )
    retained = tuple(
        replace(f, attrs={**dict(f.attrs), "specialized_from": canonical(str(f.attrs["specialized_from"]))})
        if f.attrs.get("specialized_from") in replacements else f
        for f in module.functions if f.name not in replacements
    )
    retained_points = tuple(p for p in module.selection_points if p.owner not in removed_nodes)
    point_ids = {p.id for p in retained_points}
    history = tuple(module.metadata.get("function_variant_merges", ())) + tuple(
        {"variant": variant, "canonical": canonical(variant), "reason": "equivalent_finalized_contract"}
        for variant in replacements
    )
    return verify_module(replace(
        module, nodes=nodes, functions=retained,
        selection_points=retained_points,
        selections=tuple(r for r in module.selections if r.point_id in point_ids),
        metadata={**dict(module.metadata), "function_variant_merges": history},
    ))


def _equivalent_nodes(module, source: Function, variant: Function, owners, points, records):
    if len(source.parameters) != len(variant.parameters) or len(source.outputs) != len(variant.outputs):
        return None
    if _without(source.attrs, _FUNCTION_HISTORY) != _without(variant.attrs, _FUNCTION_HISTORY):
        return None
    node_map = module.node_map
    mapping: dict[str, str] = {}
    inverse: dict[str, str] = {}
    pending = [*zip(variant.parameters, source.parameters), *zip(variant.outputs, source.outputs)]
    while pending:
        left_id, right_id = pending.pop()
        if left_id in mapping:
            if mapping[left_id] != right_id:
                return None
            continue
        if right_id in inverse:
            return None
        if left_id != right_id and (owners[left_id] != {variant.name} or owners[right_id] != {source.name}):
            return None
        left, right = node_map[left_id], node_map[right_id]
        if left_id != right_id and left.op == "builtin.var" and left_id not in variant.parameters:
            # A free variable is a captured value, not an alpha-renamable binder.
            return None
        if left.op != right.op or left.type != right.type or left.effect != right.effect or len(left.inputs) != len(right.inputs):
            return None
        mapping[left_id], inverse[right_id] = right_id, left_id
        pending.extend(zip(left.inputs, right.inputs))
    # Parameters are positional binders; even unused or same-typed parameters
    # cannot be permuted by body traversal.
    if any(mapping[left] != right for left, right in zip(variant.parameters, source.parameters)):
        return None
    for left_id, right_id in mapping.items():
        left, right = node_map[left_id], node_map[right_id]
        parameter = left_id in variant.parameters
        left_attrs = _without(left.attrs, {"name"}) if parameter else left.attrs
        right_attrs = _without(right.attrs, {"name"}) if parameter else right.attrs
        if left_attrs != right_attrs:
            return None
        left_meta = _without(left.metadata, _NODE_HISTORY)
        right_meta = _without(right.metadata, _NODE_HISTORY)
        # This parameter marker only prevents replanning within the now
        # completed boundary-layout fixed point. All other layout markers,
        # including raw restores on body expressions, remain significant.
        if parameter:
            for metadata in (left_meta, right_meta):
                if metadata.get("boundary_layout") == "distributed":
                    metadata.pop("boundary_layout")
        if left_meta != right_meta and _remap(left_meta, mapping) != right_meta:
            return None
        left_points, right_points = points[left_id], points[right_id]
        if len(left_points) != len(right_points):
            return None
        for lp, rp in zip(left_points, right_points):
            if replace(lp, id=rp.id, owner=rp.owner) != rp:
                return None
            lr, rr = records.get(lp.id), records.get(rp.id)
            if (replace(lr, point_id=rp.id) if lr is not None else None) != rr:
                return None
    return mapping


def _without(values, excluded):
    return {key: value for key, value in values.items() if key not in excluded}


def _remap(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, Mapping):
        return {key: _remap(item, mapping) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_remap(item, mapping) for item in value)
    return value


__all__ = ["merge_equivalent_function_variants"]
