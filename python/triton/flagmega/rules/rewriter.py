# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bottom-up dataflow rewriting for immutable FlagMega modules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from types import MappingProxyType

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, verify_module
from triton.flagmega.pattern_match.call_pattern import CallPattern
from triton.flagmega.pattern_match.op_pattern import OpPattern
from triton.flagmega.pattern_match.or_pattern import OrPattern
from triton.flagmega.pattern_match.unary_chain_pattern import UnaryChainPattern
from triton.flagmega.rules.core import (
    RewriteEffectPolicy,
    RewriteRedirect,
    RewriteResult,
    RewriteRule,
)


class DataflowRewriter:
    """Apply ordered local rules to a fixed point over topological IR.

    Rules may replace one node or insert helper nodes immediately before it.
    Root ids are intentionally stable, matching nncase's expression rewrite
    behavior while preserving external references in editable checkpoints.
    """

    def __init__(
        self,
        rules: Iterable[RewriteRule],
        *,
        max_iterations: int = 32,
        remove_unused: bool = True,
        rewrite_constants: bool = True,
    ) -> None:
        self.rules = tuple(rules)
        if max_iterations <= 0:
            raise ValueError("DataflowRewriter max_iterations must be positive.")
        self.max_iterations = max_iterations
        self.remove_unused = remove_unused
        self.rewrite_constants = rewrite_constants

    def rewrite(self, module: IRModule) -> IRModule:
        from triton.flagmega.passes.constants import require_constants_open

        if self.rewrite_constants:
            require_constants_open(module, "DataflowRewriter")
        current = verify_module(module)
        for _ in range(self.max_iterations):
            result = self._rewrite_sweep(current)
            if result is None:
                return _remove_unused(current) if self.remove_unused else current
            if not self.rewrite_constants:
                from triton.flagmega.rules.region import verify_opaque_constants
                verify_opaque_constants(current, result)
            current = verify_module(result)
        raise IRVerificationError(
            f"Dataflow rewrite did not converge after {self.max_iterations} iterations.",
            stage=current.stage,
        )

    def _rewrite_sweep(self, module: IRModule) -> IRModule | None:
        """Rewrite every root present at sweep entry at most once."""

        current = module
        changed = False
        positions: dict[str, tuple[int, int]] | None = None
        offset = 0
        roots = tuple((rule.root_ops if rule.root_ops is not None else _root_operators(rule.pattern), rule)
                      for rule in self.rules)
        dispatch = {}
        users = [None]
        for original in module.nodes:
            node_id = original.id
            node = current.node_map.get(node_id)
            if node is None:
                continue
            op = node.op
            if op not in dispatch:
                dispatch[op] = tuple(rule for operators, rule in roots if operators is None or op in operators)
            if not dispatch[op]:
                continue
            if positions is None:
                positions = {node.id: (index, 0) for index, node in enumerate(current.nodes)}
                offset = 0
            rank = positions.get(node_id)
            if rank is None:
                continue
            index = rank[0] + offset
            rewritten = self._rewrite_node(current, index, positions, dispatch[op], users)
            if rewritten is not None:
                current, delta = rewritten
                changed = True
                # Local edits shift every later original root equally.
                # Region edits may remove future roots and require a rebase.
                if delta is None:
                    positions = None
                else:
                    offset += delta
        return current if changed else None

    def _rewrite_node(self, module: IRModule, index: int, positions, rules, users) -> tuple[IRModule, int | None] | None:
        node = module.nodes[index]
        rank = positions[node.id]
        if not self.rewrite_constants and node.op == "builtin.const_asset":
            return None
        for rule in rules:
            rewritten = rule.apply(node, module)
            if rewritten is None:
                continue
            if isinstance(rewritten, RewriteRedirect):
                target = module.node_map.get(rewritten.target_id)
                if target is None or target.type != node.type:
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} redirect must preserve root type.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                if (
                    not node.effect.is_pure
                    and rule.effect_policy is not RewriteEffectPolicy.ALLOW
                ):
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} cannot redirect effectful node without "
                        "effect_policy=ALLOW.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                if positions[rewritten.target_id] >= rank:
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} redirects to non-preceding node {rewritten.target_id!r}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                result = _redirect(module, node.id, rewritten.target_id, index, positions, users)
                positions.pop(node.id)
                return result, -1
            result = rewritten if isinstance(rewritten, RewriteResult) else RewriteResult(rewritten)
            if result.replacement.id != node.id:
                raise IRVerificationError(
                    f"Rewrite rule {rule.name!r} must preserve root id {node.id!r}, "
                    f"got {result.replacement.id!r}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            if not result.prefix_nodes and not result.removed_ids and not result.extra_replacements and result.replacement == node:
                continue
            if rule.effect_policy is RewriteEffectPolicy.PRESERVE:
                if result.replacement.effect != node.effect:
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} must preserve root effect {node.effect}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                effectful_helpers = tuple(
                    helper.id for helper in result.prefix_nodes if not helper.effect.is_pure
                )
                if effectful_helpers:
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} inserted effectful helpers "
                        f"{effectful_helpers} without effect_policy=ALLOW.",
                        stage=module.stage,
                        node_id=node.id,
                    )
            if result.removed_ids or result.extra_replacements or result.insertion_before is not None:
                from triton.flagmega.rules.region import apply_region
                users[0] = None
                return apply_region(module, node, result, rule), None
            occupied = module.node_map.keys()
            prefix_ids = [value.id for value in result.prefix_nodes]
            collisions = occupied & set(prefix_ids)
            if collisions or len(prefix_ids) != len(set(prefix_ids)):
                raise IRVerificationError(
                    f"Rewrite rule {rule.name!r} generated colliding helper ids {sorted(collisions)}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            helpers = set()

            def missing_inputs(value, helpers=helpers):
                return {input_id for input_id in value.inputs
                        if input_id not in helpers and positions.get(input_id, rank) >= rank}

            for helper in result.prefix_nodes:
                missing = missing_inputs(helper)
                if missing:
                    raise IRVerificationError(
                        f"Rewrite helper {helper.id!r} has non-topological inputs {sorted(missing)}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                helpers.add(helper.id)
            missing = missing_inputs(result.replacement)
            if missing:
                raise IRVerificationError(
                    f"Rewrite replacement {node.id!r} has non-topological inputs {sorted(missing)}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            nodes = (
                *module.nodes[:index],
                *result.prefix_nodes,
                result.replacement,
                *module.nodes[index + 1:],
            )
            updated = replace(module, nodes=nodes)
            # Copy the existing immutable index in C and update only changed
            # entries. Rebuilding it in Python after every local rewrite made
            # sparse edits quadratic in the complete weight-expression graph.
            node_map = module.node_map.copy()
            node_map.update((value.id, value) for value in result.prefix_nodes)
            node_map[node.id] = result.replacement
            object.__setattr__(updated, "node_map", MappingProxyType(node_map))
            if users[0] is not None:
                for input_id in set(node.inputs):
                    users[0][input_id].discard(node.id)
                for value in (*result.prefix_nodes, result.replacement):
                    for input_id in set(value.inputs):
                        users[0].setdefault(input_id, set()).add(value.id)
            # Helpers have an earlier order than their anchor. Their relative
            # order was checked above; later roots only need that relation.
            positions.update((value.id, (rank[0], -1)) for value in result.prefix_nodes)
            return updated, len(result.prefix_nodes)
        return None


def _root_operators(pattern):
    """Necessary root names only; custom and predicate patterns stay generic."""
    if type(pattern) is OpPattern:
        return frozenset((pattern.op_name,))
    if type(pattern) is CallPattern and type(pattern.target) is OpPattern:
        return frozenset((pattern.target.op_name,))
    if type(pattern) in {OrPattern, UnaryChainPattern}:
        branches = ((pattern.lhs, pattern.rhs) if type(pattern) is OrPattern
                    else (pattern.step, pattern.terminal))
        left, right = (_root_operators(branch) for branch in branches)
        return None if left is None or right is None else left | right
    return None


def _redirect(module: IRModule, source: str, target: str, index=None, positions=None, users=None) -> IRModule:
    # A redirect edits direct users only. Build the inverse relation once per
    # sweep and maintain it for subsequent local edits; sparse redirects must
    # not repeatedly inspect every unrelated weight expression.
    if positions is None:
        positions = {node.id: (ordinal, 0) for ordinal, node in enumerate(module.nodes)}
    if index is None:
        index = positions[source][0]
    if users is None:
        users = [None]
    if users[0] is None:
        users[0] = {}
        for node in module.nodes:
            for input_id in set(node.inputs):
                users[0].setdefault(input_id, set()).add(node.id)
    readers = users[0].pop(source, set())
    nodes = list(module.nodes)
    node_map = module.node_map.copy()
    offset = index - positions[source][0]
    for node_id in readers:
        node = node_map[node_id]
        updated = replace(node, inputs=tuple(target if value == source else value for value in node.inputs))
        nodes[positions[node_id][0] + offset] = updated
        node_map[node_id] = updated
    users[0].setdefault(target, set()).update(readers)
    for input_id in set(node_map[source].inputs):
        users[0][input_id].discard(source)
    nodes.pop(index)
    node_map.pop(source)
    functions = tuple(
        replace(
            function,
            parameters=tuple(target if value == source else value for value in function.parameters),
            outputs=tuple(target if value == source else value for value in function.outputs),
        )
        if source in function.parameters or source in function.outputs
        else function
        for function in module.functions
    )
    points = tuple(replace(point, owner=target) if point.owner == source else point for point in module.selection_points)
    result = replace(module, nodes=tuple(nodes), functions=functions, selection_points=points)
    object.__setattr__(result, "node_map", MappingProxyType(node_map))
    return result


def _remove_unused(module: IRModule) -> IRModule:
    required = {
        node_id
        for function in module.functions
        for node_id in (*function.parameters, *function.outputs)
    }
    required.update(node.id for node in module.nodes if not node.effect.is_pure)
    node_map = module.node_map
    pending = list(required)
    while pending:
        node_id = pending.pop()
        for input_id in node_map[node_id].inputs:
            if input_id not in required:
                required.add(input_id)
                pending.append(input_id)
    if len(required) == len(module.nodes):
        return module
    representatives: dict[str, str] = {}
    for node in module.nodes:
        if node.id not in required:
            continue
        for key in ("selection_owner_for", "vectorization_root"):
            if key in node.metadata:
                # The last surviving node is the outermost representative in
                # topological order.  This preserves the pre-existing
                # vectorization-owner contract when a rewrite introduces
                # several helpers for the same logical root.
                representatives[str(node.metadata[key])] = node.id
    points = tuple(
        replace(point, owner=representatives[point.owner])
        if point.owner is not None and point.owner not in required and point.owner in representatives
        else point
        for point in module.selection_points
        if point.owner is None or point.owner in required or point.owner in representatives
    )
    point_ids = {point.id for point in points}
    return replace(
        module,
        nodes=tuple(node for node in module.nodes if node.id in required),
        selection_points=points,
        selections=tuple(record for record in module.selections if record.point_id in point_ids),
    )


__all__ = ["DataflowRewriter"]
