# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Transactional, topologically checked multi-result dataflow rewrites."""

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import verify_module
from triton.flagmega.rules.core import RewriteEffectPolicy


def apply_region(module, root, result, rule):

    def fail(message):
        raise IRVerificationError(f"Rewrite rule {rule.name!r}: {message}", stage=module.stage, node_id=root.id)

    nodes = module.node_map
    removed = set(result.removed_ids)
    extras = {node.id: node for node in result.extra_replacements}
    helpers = [node.id for node in result.prefix_nodes]
    if root.id in removed or root.id in extras or removed & extras.keys():
        fail("root, removed ids and extra replacements must be disjoint.")
    if len(removed) != len(result.removed_ids) or len(extras) != len(result.extra_replacements):
        fail("duplicate region ids.")
    if (removed | extras.keys()) - nodes.keys():
        fail("region references absent nodes.")
    if len(set(helpers)) != len(helpers) or set(helpers) & (nodes.keys() - removed):
        fail("colliding helper ids; relocation requires explicit removal.")
    for node in extras.values():
        if node.type != nodes[node.id].type:
            fail("extra replacement must preserve its boundary type.")
    if result.replacement.type != root.type:
        fail("region replacement must preserve root type.")
    if rule.effect_policy is RewriteEffectPolicy.PRESERVE:
        if any(not nodes[key].effect.is_pure for key in removed):
            fail("effectful removal requires effect_policy=ALLOW.")
        if any(node.effect != nodes[node.id].effect for node in extras.values()):
            fail("extra replacement changed effects without effect_policy=ALLOW.")

    rewritten = []
    anchor = result.insertion_before or root.id
    if anchor not in nodes or anchor in removed:
        fail("helper insertion boundary must survive the rewrite.")
    for node in module.nodes:
        if node.id == anchor:
            rewritten.extend(result.prefix_nodes)
        if node.id == root.id:
            rewritten.append(result.replacement)
        elif node.id not in removed:
            rewritten.append(extras.get(node.id, node))
    available = set()
    for node in rewritten:
        missing = set(node.inputs) - available
        if missing:
            fail(f"non-topological inputs for {node.id!r}: {sorted(missing)}.")
        available.add(node.id)
    for function in module.functions:
        if set((*function.parameters, *function.outputs)) - available:
            fail("region removed a function boundary.")
    changed = removed | {root.id} | extras.keys()
    points = tuple(point for point in module.selection_points if point.owner not in changed)
    point_ids = {point.id for point in points}
    return verify_module(
        replace(module, nodes=tuple(rewritten), selection_points=points,
                selections=tuple(record for record in module.selections if record.point_id in point_ids)))


def verify_opaque_constants(before, after):
    """Frozen recipes are read-only leaves, never a late rewrite substrate."""
    for node in after.nodes:
        if node.op == "builtin.const_asset" or (node.id in before.node_map
                                                and before.node_map[node.id].op == "builtin.const_asset"):
            if before.node_map.get(node.id) != node:
                raise IRVerificationError("Dataflow rewrite cannot modify opaque constant assets.", stage=before.stage,
                                          node_id=node.id)
    if before.constant_recipes != after.constant_recipes:
        raise IRVerificationError("Dataflow rewrite cannot modify opaque constant recipes.", stage=before.stage)
