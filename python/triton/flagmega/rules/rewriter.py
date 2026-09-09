# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bottom-up dataflow rewriting for immutable FlagMega modules."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, verify_module
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
        positions: dict[str, int] | None = None
        for node_id in tuple(node.id for node in module.nodes):
            if positions is None:
                positions = {node.id: index for index, node in enumerate(current.nodes)}
            index = positions.get(node_id)
            if index is None:
                continue
            rewritten = self._rewrite_node(current, index)
            if rewritten is not None:
                current = rewritten
                changed = True
                # Helpers/redirects can shift later roots. Keep the sweep's
                # original visit order but rebuild offsets for the new snapshot.
                positions = None
        return current if changed else None

    def _rewrite_node(self, module: IRModule, index: int) -> IRModule | None:
        node = module.nodes[index]
        if not self.rewrite_constants and node.op == "builtin.const_asset":
            return None
        for rule in self.rules:
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
                if rewritten.target_id not in {value.id for value in module.nodes[:index]}:
                    raise IRVerificationError(
                        f"Rewrite rule {rule.name!r} redirects to non-preceding node {rewritten.target_id!r}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                return _redirect(module, node.id, rewritten.target_id)
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
                return apply_region(module, node, result, rule)
            occupied = {value.id for value in module.nodes}
            prefix_ids = [value.id for value in result.prefix_nodes]
            collisions = occupied.intersection(prefix_ids)
            if collisions or len(prefix_ids) != len(set(prefix_ids)):
                raise IRVerificationError(
                    f"Rewrite rule {rule.name!r} generated colliding helper ids {sorted(collisions)}.",
                    stage=module.stage,
                    node_id=node.id,
                )
            available = {value.id for value in module.nodes[:index]}
            for helper in result.prefix_nodes:
                missing = set(helper.inputs) - available
                if missing:
                    raise IRVerificationError(
                        f"Rewrite helper {helper.id!r} has non-topological inputs {sorted(missing)}.",
                        stage=module.stage,
                        node_id=node.id,
                    )
                available.add(helper.id)
            missing = set(result.replacement.inputs) - available
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
            return replace(module, nodes=nodes)
        return None


def _redirect(module: IRModule, source: str, target: str) -> IRModule:
    nodes = tuple(
        replace(node, inputs=tuple(target if value == source else value for value in node.inputs))
        for node in module.nodes
        if node.id != source
    )
    functions = tuple(replace(
        function,
        parameters=tuple(target if value == source else value for value in function.parameters),
        outputs=tuple(target if value == source else value for value in function.outputs),
    ) for function in module.functions)
    points = tuple(replace(point, owner=target) if point.owner == source else point for point in module.selection_points)
    return replace(module, nodes=nodes, functions=functions, selection_points=points)


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
