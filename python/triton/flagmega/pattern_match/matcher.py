# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Deterministic data-flow matcher for FlagMega expression IR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import weakref

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import IRModule, Node
from triton.flagmega.pattern_match.call_pattern import CallPattern
from triton.flagmega.pattern_match.op_pattern import OpPattern
from triton.flagmega.pattern_match.or_pattern import OrPattern
from triton.flagmega.pattern_match.options import MatchOptions
from triton.flagmega.pattern_match.pattern import Pattern
from triton.flagmega.pattern_match.result import MatchResult
from triton.flagmega.pattern_match.vargs_pattern import VArgsPattern


@dataclass
class _MatchScope:
    matches: dict[Pattern, object] = field(default_factory=dict)

    def clone(self) -> _MatchScope:
        return _MatchScope(dict(self.matches))

    def capture(self, pattern: Pattern, value: object) -> bool:
        previous = self.matches.get(pattern, _MISSING)
        if previous is not _MISSING:
            return _same_value(previous, value)
        self.matches[pattern] = value
        return True


_MISSING = object()
_USER_COUNTS: dict[int, tuple[weakref.ReferenceType[IRModule], dict[str, int]]] = {}


def try_match_root(
    node: Node,
    pattern: Pattern,
    module: IRModule,
    options: MatchOptions | None = None,
) -> MatchResult | None:
    """Match ``pattern`` at exactly ``node`` and return named captures."""

    if node.id not in module.node_map:
        raise ValueError(f"Node {node.id!r} does not belong to the supplied module.")
    options = options or MatchOptions()
    if options.is_suppressed(node, pattern):
        return None
    scope = _MatchScope()
    if not _match(node, pattern, module, scope, options):
        return None
    return MatchResult(node, scope.matches)


def try_match(
    module: IRModule,
    pattern: Pattern,
    options: MatchOptions | None = None,
) -> MatchResult | None:
    """Find the first match, preferring graph outputs and later nodes."""

    for node in _output_first_nodes(module):
        result = try_match_root(node, pattern, module, options)
        if result is not None:
            return result
    return None


def find_matches(
    module: IRModule,
    pattern: Pattern,
    options: MatchOptions | None = None,
) -> tuple[MatchResult, ...]:
    reachable = {node.id for node in _output_first_nodes(module)}
    return tuple(
        result
        for node in module.nodes
        if node.id in reachable
        and (result := try_match_root(node, pattern, module, options)) is not None
    )


def _match(
    node: Node,
    pattern: Pattern,
    module: IRModule,
    scope: _MatchScope,
    options: MatchOptions,
) -> bool:
    if options.is_suppressed(node, pattern):
        return False
    if pattern.user_count is not None and _cached_user_counts(module).get(node.id, 0) != pattern.user_count:
        return False
    previous = scope.matches.get(pattern, _MISSING)
    if previous is not _MISSING:
        return _same_value(previous, node)

    if isinstance(pattern, OrPattern):
        for alternative in (pattern.lhs, pattern.rhs):
            branch = scope.clone()
            if _match(node, alternative, module, branch, options) and branch.capture(pattern, node):
                scope.matches = branch.matches
                return True
        return False

    if isinstance(pattern, CallPattern):
        return _match_call(node, pattern, module, scope, options)

    if isinstance(pattern, VArgsPattern):
        return False

    if not pattern.match_leaf(node):
        return False
    return scope.capture(pattern, node)


def _match_call(
    node: Node,
    pattern: CallPattern,
    module: IRModule,
    scope: _MatchScope,
    options: MatchOptions,
) -> bool:
    if not pattern.match_leaf(node):
        return False
    operands = tuple(module.node_map[node_id] for node_id in node.inputs)
    try:
        from triton.flagmega.ir.ops.core import get_definition

        definition = get_definition(node.op)
        fixed = sum(not parameter.variadic for parameter in definition.input_parameters)
        variadic = any(parameter.variadic for parameter in definition.input_parameters)
        if len(operands) < fixed or (not variadic and len(operands) != fixed):
            return False
        definition.verify_parameter_types(operands)
    except (IRSchemaError, KeyError):
        return False
    operand_patterns = pattern.arguments.patterns_for(operands)
    if operand_patterns is None:
        return False

    branch = scope.clone()
    if not _match(node, pattern.target, module, branch, options):
        return False
    for operand, operand_pattern in zip(operands, operand_patterns):
        if not _match(operand, operand_pattern, module, branch, options):
            return False
    if not branch.capture(pattern.arguments, operands):
        return False
    if not branch.capture(pattern, node):
        return False
    scope.matches = branch.matches
    return True


def _same_value(lhs: object, rhs: object) -> bool:
    if isinstance(lhs, Node) and isinstance(rhs, Node):
        return lhs.id == rhs.id
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return len(lhs) == len(rhs) and all(_same_value(a, b) for a, b in zip(lhs, rhs))
    return lhs == rhs


def _user_counts(module: IRModule) -> dict[str, int]:
    users: dict[str, set[str]] = {node.id: set() for node in module.nodes}
    for node in module.nodes:
        for input_id in node.inputs:
            users[input_id].add(f"node:{node.id}")
    for function in module.functions:
        for index, output in enumerate(function.outputs):
            users[output].add(f"function:{function.name}:{index}")
    return {node_id: len(values) for node_id, values in users.items()}


def _cached_user_counts(module: IRModule) -> dict[str, int]:
    """Compute Users only when a pattern asks, once per immutable snapshot."""

    identity = id(module)
    cached = _USER_COUNTS.get(identity)
    if cached is not None and cached[0]() is module:
        return cached[1]
    counts = _user_counts(module)

    def discard(reference: weakref.ReferenceType[IRModule]) -> None:
        current = _USER_COUNTS.get(identity)
        if current is not None and current[0] is reference:
            _USER_COUNTS.pop(identity, None)

    _USER_COUNTS[identity] = (weakref.ref(module, discard), counts)
    return counts


def _output_first_nodes(module: IRModule) -> tuple[Node, ...]:
    order = []
    seen = set()
    node_map = module.node_map

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        node = node_map[node_id]
        order.append(node)
        for input_id in reversed(node.inputs):
            visit(input_id)

    functions = (
        module.function_map[module.entry],
        *(function for function in module.functions if function.name != module.entry),
    )
    for function in functions:
        for output in reversed(function.outputs):
            visit(output)
    return tuple(order)


__all__ = ["find_matches", "try_match", "try_match_root"]
