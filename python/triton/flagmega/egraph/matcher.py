# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Structural pattern matching over e-classes, aligned with nncase EGraphMatcher."""

from __future__ import annotations

from dataclasses import dataclass, field

from triton.flagmega.egraph.graph import EGraph, ENode
from triton.flagmega.ir.model import IRModule, Node
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.pattern_match.call_pattern import CallPattern
from triton.flagmega.pattern_match.or_pattern import OrPattern
from triton.flagmega.pattern_match.pattern import Pattern
from triton.flagmega.pattern_match.result import MatchResult
from triton.flagmega.pattern_match.vargs_pattern import VArgsPattern
from triton.flagmega.pattern_match.unary_chain_pattern import UnaryChainPattern


@dataclass
class _Scope:
    matches: dict[Pattern, object] = field(default_factory=dict)
    user_counts: dict[str, int] = field(default_factory=dict)

    def clone(self) -> _Scope:
        return _Scope(dict(self.matches), self.user_counts)

    def capture(self, pattern: Pattern, value: object) -> bool:
        previous = self.matches.get(pattern, _MISSING)
        if previous is not _MISSING:
            return _same_identity(previous, value)
        self.matches[pattern] = value
        return True


_MISSING = object()


def find_egraph_matches(
    graph: EGraph,
    pattern: Pattern,
    module: IRModule,
) -> tuple[MatchResult, ...]:
    """Return every match, exploring alternatives in every child e-class."""

    user_counts = _user_counts(module)
    results: list[MatchResult] = []
    seen: set[tuple[object, ...]] = set()
    # Snapshot order is deterministic and rules cannot mutate this traversal.
    for _class_id, enode in graph.enodes():
        if enode.op == "egraph.opaque":
            continue
        for scope in _match_enode(graph, enode, pattern, _Scope(user_counts=user_counts), module):
            signature = _scope_signature(enode.node, scope.matches)
            if signature in seen:
                continue
            seen.add(signature)
            results.append(MatchResult(enode.node, scope.matches))
    return tuple(results)


def _match_class(
    graph: EGraph,
    class_id: int,
    pattern: Pattern,
    scope: _Scope,
    module: IRModule,
) -> list[_Scope]:
    results: list[_Scope] = []
    for enode in graph.class_view(class_id).nodes:
        results.extend(_match_enode(graph, enode, pattern, scope, module))
    return results


def _match_enode(
    graph: EGraph,
    enode: ENode,
    pattern: Pattern,
    scope: _Scope,
    module: IRModule,
) -> list[_Scope]:
    node = enode.node
    if pattern.user_count is not None and scope.user_counts.get(node.id, 0) != pattern.user_count:
        return []
    previous = scope.matches.get(pattern, _MISSING)
    if previous is not _MISSING and not isinstance(pattern, UnaryChainPattern):
        return [scope] if _same_identity(previous, node) else []

    if isinstance(pattern, UnaryChainPattern):
        results = []
        pending = [(enode, ())]
        while pending:
            current, path = pending.pop()
            for branch in _match_enode(graph, current, pattern.terminal, scope.clone(), module):
                if branch.capture(pattern, tuple(value.node for value in path)):
                    results.append(branch)
            if (len(current.children) != 1 or any(value.node.id == current.node.id for value in path)
                    or not _match_enode(graph, current, pattern.step,
                                       _Scope(user_counts=scope.user_counts), module)):
                continue
            pending.extend((child, (*path, current)) for child in
                           graph.class_view(current.children[0]).nodes)
        return results

    if isinstance(pattern, OrPattern):
        results: list[_Scope] = []
        for alternative in (pattern.lhs, pattern.rhs):
            for branch in _match_enode(graph, enode, alternative, scope.clone(), module):
                if branch.capture(pattern, node):
                    results.append(branch)
        return results

    if isinstance(pattern, CallPattern):
        return _match_call(graph, enode, pattern, scope, module)

    if isinstance(pattern, VArgsPattern) or not pattern.match_leaf(node):
        return []
    branch = scope.clone()
    return [branch] if branch.capture(pattern, node) else []


def _match_call(
    graph: EGraph,
    enode: ENode,
    pattern: CallPattern,
    scope: _Scope,
    module: IRModule,
) -> list[_Scope]:
    node = enode.node
    if not pattern.match_leaf(node):
        return []
    try:
        definition = get_definition(node.op)
    except KeyError:
        return []
    parameters = definition.input_parameters
    fixed = sum(not parameter.variadic for parameter in parameters)
    variadic = parameters[-1] if parameters and parameters[-1].variadic else None
    if len(enode.children) < fixed or (variadic is None and len(enode.children) != fixed):
        return []
    for index, child in enumerate(enode.children):
        parameter = parameters[index] if index < fixed else variadic
        if parameter is None or not parameter.type_pattern.match_leaf(graph.class_view(child).type):
            return []

    node_map = module.node_map
    operands = tuple(
        node_map.get(input_id, graph.class_view(child).nodes[0].node)
        for input_id, child in zip(node.inputs, enode.children)
    )
    operand_patterns = pattern.arguments.patterns_for(operands)
    if operand_patterns is None:
        return []

    scopes = _match_enode(graph, enode, pattern.target, scope.clone(), module)
    for child, operand_pattern in zip(enode.children, operand_patterns):
        next_scopes: list[_Scope] = []
        for candidate in scopes:
            next_scopes.extend(_match_class(graph, child, operand_pattern, candidate, module))
        scopes = next_scopes
        if not scopes:
            return []

    results: list[_Scope] = []
    for branch in scopes:
        arguments = tuple(branch.matches[value] for value in operand_patterns)
        if branch.capture(pattern.arguments, arguments) and branch.capture(pattern, node):
            results.append(branch)
    return results


def _same_identity(lhs: object, rhs: object) -> bool:
    if isinstance(lhs, Node) and isinstance(rhs, Node):
        # nncase's e-graph match memo uses expression reference identity.  Two
        # alternatives deliberately share the stable source id, so id-string
        # equality would incorrectly conflate different expressions.
        return lhs is rhs
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return len(lhs) == len(rhs) and all(
            _same_identity(left, right) for left, right in zip(lhs, rhs)
        )
    return lhs is rhs or lhs == rhs


def _user_counts(module: IRModule) -> dict[str, int]:
    users: dict[str, set[str]] = {node.id: set() for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            users.setdefault(input_id, set()).add(f"node:{node.id}")
    for function in module.functions:
        for index, output in enumerate(function.outputs):
            users.setdefault(output, set()).add(f"function:{function.name}:{index}")
    return {node_id: len(values) for node_id, values in users.items()}


def _scope_signature(root: Node, matches: dict[Pattern, object]) -> tuple[object, ...]:
    def identity(value: object) -> object:
        if isinstance(value, Node):
            return ("node", id(value))
        if isinstance(value, tuple):
            return ("tuple", *(identity(item) for item in value))
        return ("value", type(value).__qualname__, repr(value))

    return (
        id(root),
        *(item for pattern in sorted(matches, key=id) for item in (id(pattern), identity(matches[pattern]))),
    )


__all__ = ["find_egraph_matches"]
