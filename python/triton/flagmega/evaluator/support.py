# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Static evaluator-handler coverage for executable module regions."""

from __future__ import annotations

from dataclasses import dataclass

from triton.flagmega.errors import EvaluationError
from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.ops.core import OpDefinition, get_definition


_VISITOR_HANDLED_OPS = frozenset({"builtin.call"})


@dataclass(frozen=True)
class EvaluationHandlerGap:
    node_id: str
    op: str
    owner: str


@dataclass(frozen=True)
class EvaluationSupport:
    """Coverage report for every node that an evaluator run can execute."""

    supported_ops: tuple[str, ...]
    visitor_handled_ops: tuple[str, ...]
    gaps: tuple[EvaluationHandlerGap, ...]

    @property
    def is_complete(self) -> bool:
        return not self.gaps

    def require_complete(self, *, stage: str | None = None) -> None:
        if self.is_complete:
            return
        details = ", ".join(
            f"{gap.node_id}:{gap.op} ({gap.owner})" for gap in self.gaps
        )
        raise EvaluationError(
            f"Reference evaluator has no handler for executable nodes: {details}.",
            stage=stage,
            node_id=self.gaps[0].node_id,
        )


def inspect_evaluation_support(module: IRModule) -> EvaluationSupport:
    """Inspect live entry/callee nodes and all eagerly materialized recipes."""

    owned = executable_nodes(module)
    supported: set[str] = set()
    visitor_handled: set[str] = set()
    gaps: list[EvaluationHandlerGap] = []
    for node, owner in owned:
        if node.op in _VISITOR_HANDLED_OPS:
            visitor_handled.add(node.op)
            continue
        definition = get_definition(node.op)
        if _has_evaluate_handler(definition):
            supported.add(node.op)
        else:
            gaps.append(EvaluationHandlerGap(node.id, node.op, owner))
    return EvaluationSupport(
        tuple(sorted(supported)),
        tuple(sorted(visitor_handled)),
        tuple(gaps),
    )


def executable_nodes(module: IRModule) -> tuple[tuple[Node, str], ...]:
    """Return nodes reachable by a reference run with their owning region."""

    live = _live_function_nodes(module)
    owned: list[tuple[Node, str]] = [
        (node, "function") for node in module.nodes if node.id in live
    ]
    owned.extend(
        (node, f"constant recipe {recipe.id}")
        for recipe in module.constant_recipes
        for node in recipe.nodes
    )
    return tuple(owned)


def _has_evaluate_handler(definition: type[OpDefinition]) -> bool:
    handler = getattr(definition.evaluate, "__func__", definition.evaluate)
    base = getattr(OpDefinition.evaluate, "__func__", OpDefinition.evaluate)
    return handler is not base


def _live_function_nodes(module: IRModule) -> set[str]:
    node_map = module.node_map
    function_map = module.function_map
    live: set[str] = set()
    visited_functions: set[str] = set()
    pending_functions = [module.entry]
    while pending_functions:
        function_name = pending_functions.pop()
        if function_name in visited_functions:
            continue
        visited_functions.add(function_name)
        function = function_map[function_name]
        pending_nodes = list((*function.parameters, *function.outputs))
        while pending_nodes:
            node_id = pending_nodes.pop()
            if node_id in live:
                continue
            node = node_map[node_id]
            live.add(node_id)
            pending_nodes.extend(node.inputs)
            if node.op == "builtin.call":
                pending_functions.append(str(node.attrs["callee"]))
    return live


__all__ = [
    "EvaluationHandlerGap",
    "EvaluationSupport",
    "executable_nodes",
    "inspect_evaluation_support",
]
