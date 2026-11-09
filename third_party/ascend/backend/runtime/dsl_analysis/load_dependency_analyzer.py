# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple


def _is_tl_load_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "load")


def _expr_has_tl_load(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if _is_tl_load_call(node):
            return True
    return False


def _extract_target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for elt in target.elts:
            names.extend(_extract_target_names(elt))
        return names
    return []


def _find_function_by_name(tree: ast.AST, function_name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == function_name:
            return node
    return None


def _iter_assignments(tree: ast.AST) -> Iterable[Tuple[List[str], ast.AST]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if node.value is None:
                continue
            target_names: List[str] = []
            for target in node.targets:
                target_names.extend(_extract_target_names(target))
            if target_names:
                yield target_names, node.value
            continue

        if isinstance(node, ast.AnnAssign):
            if node.value is None:
                continue
            target_names = _extract_target_names(node.target)
            if target_names:
                yield target_names, node.value
            continue

        if isinstance(node, ast.NamedExpr):
            target_names = _extract_target_names(node.target)
            if target_names:
                yield target_names, node.value


def _expr_uses_symbols(expr: ast.AST, symbols: Set[str]) -> bool:
    if not symbols:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in symbols:
            return True
    return False


def _legacy_collect_load_derived_symbols(tree: ast.AST) -> Set[str]:
    assignments = list(_iter_assignments(tree))
    load_derived: Set[str] = set()

    for target_names, value in assignments:
        if not _is_tl_load_call(value):
            continue
        for name in target_names:
            load_derived.add(name)

    changed = True
    while changed:
        changed = False
        for target_names, value in assignments:
            if not _expr_uses_symbols(value, load_derived):
                continue
            for name in target_names:
                if name in load_derived:
                    continue
                load_derived.add(name)
                changed = True

    return load_derived


def _to_static_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _resolve_static_scalar(expr: ast.AST, known_values: Mapping[str, object]) -> Optional[object]:
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, (bool, int, float)):
            return expr.value
        return None
    if isinstance(expr, ast.Name):
        value = known_values.get(expr.id, None)
        if isinstance(value, (bool, int, float)):
            return value
        return None
    if isinstance(expr, ast.UnaryOp):
        operand = _resolve_static_scalar(expr.operand, known_values)
        if operand is None:
            return None
        if isinstance(expr.op, ast.Not):
            operand_bool = _to_static_bool(operand)
            if operand_bool is None:
                return None
            return not operand_bool
        if isinstance(expr.op, ast.USub):
            if isinstance(operand, (int, float)):
                return -operand
            return None
        if isinstance(expr.op, ast.UAdd):
            if isinstance(operand, (int, float)):
                return +operand
            return None
        return None
    if isinstance(expr, ast.BoolOp):
        if isinstance(expr.op, ast.And):
            maybe_unknown = False
            for value_expr in expr.values:
                value = _resolve_static_bool_expr(value_expr, known_values)
                if value is False:
                    return False
                if value is None:
                    maybe_unknown = True
            return None if maybe_unknown else True
        if isinstance(expr.op, ast.Or):
            maybe_unknown = False
            for value_expr in expr.values:
                value = _resolve_static_bool_expr(value_expr, known_values)
                if value is True:
                    return True
                if value is None:
                    maybe_unknown = True
            return None if maybe_unknown else False
        return None
    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            return None
        lhs = _resolve_static_scalar(expr.left, known_values)
        rhs = _resolve_static_scalar(expr.comparators[0], known_values)
        if lhs is None or rhs is None:
            return None
        op = expr.ops[0]
        if isinstance(op, ast.Eq):
            return lhs == rhs
        if isinstance(op, ast.NotEq):
            return lhs != rhs
        if isinstance(op, ast.Lt):
            return lhs < rhs
        if isinstance(op, ast.LtE):
            return lhs <= rhs
        if isinstance(op, ast.Gt):
            return lhs > rhs
        if isinstance(op, ast.GtE):
            return lhs >= rhs
        return None
    return None


def _resolve_static_bool_expr(expr: ast.AST, known_values: Mapping[str, object]) -> Optional[bool]:
    scalar = _resolve_static_scalar(expr, known_values)
    if scalar is None:
        return None
    return _to_static_bool(scalar)


def _assign_target_value(
    target_names: List[str],
    value_expr: ast.AST,
    load_derived: Set[str],
) -> None:
    is_derived = _expr_has_tl_load(value_expr) or _expr_uses_symbols(value_expr, load_derived)
    if is_derived:
        for name in target_names:
            load_derived.add(name)
        return
    for name in target_names:
        load_derived.discard(name)


def _visit_stmt(
    stmt: ast.stmt,
    load_derived: Set[str],
    known_values: Mapping[str, object],
) -> None:
    if isinstance(stmt, ast.Assign):
        if stmt.value is None:
            return
        target_names: List[str] = []
        for target in stmt.targets:
            target_names.extend(_extract_target_names(target))
        if target_names:
            _assign_target_value(target_names, stmt.value, load_derived)
        return

    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return
        target_names = _extract_target_names(stmt.target)
        if target_names:
            _assign_target_value(target_names, stmt.value, load_derived)
        return

    if isinstance(stmt, ast.AugAssign):
        target_names = _extract_target_names(stmt.target)
        if not target_names:
            return
        is_derived = (_expr_has_tl_load(stmt.value) or _expr_uses_symbols(stmt.value, load_derived)
                      or any(name in load_derived for name in target_names))
        if is_derived:
            for name in target_names:
                load_derived.add(name)
        return

    if isinstance(stmt, ast.If):
        cond = _resolve_static_bool_expr(stmt.test, known_values)
        if cond is True:
            for child in stmt.body:
                _visit_stmt(child, load_derived, known_values)
            return
        if cond is False:
            for child in stmt.orelse:
                _visit_stmt(child, load_derived, known_values)
            return
        body_load_derived = set(load_derived)
        else_load_derived = set(load_derived)
        for child in stmt.body:
            _visit_stmt(child, body_load_derived, known_values)
        for child in stmt.orelse:
            _visit_stmt(child, else_load_derived, known_values)
        load_derived.clear()
        load_derived.update(body_load_derived.union(else_load_derived))
        return

    if isinstance(stmt, ast.For):
        body_load_derived = set(load_derived)
        for child in stmt.body:
            _visit_stmt(child, body_load_derived, known_values)
        for child in stmt.orelse:
            _visit_stmt(child, body_load_derived, known_values)
        load_derived.update(body_load_derived)
        return

    if isinstance(stmt, ast.While):
        body_load_derived = set(load_derived)
        for child in stmt.body:
            _visit_stmt(child, body_load_derived, known_values)
        for child in stmt.orelse:
            _visit_stmt(child, body_load_derived, known_values)
        load_derived.update(body_load_derived)
        return

    if isinstance(stmt, ast.With):
        for child in stmt.body:
            _visit_stmt(child, load_derived, known_values)
        return


def _collect_load_derived_from_function(
    function_node: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
) -> Set[str]:
    known_values: Dict[str, object] = {}
    for name, value in (provided_args or {}).items():
        if isinstance(value, (bool, int, float)):
            known_values[name] = value

    load_derived: Set[str] = set()
    body = getattr(function_node, "body", None)
    if not isinstance(body, list):
        return load_derived

    for stmt in body:
        if isinstance(stmt, ast.stmt):
            _visit_stmt(stmt, load_derived, known_values)
    return load_derived


def collect_load_derived_symbols(
    tree: ast.AST,
    *,
    provided_args: Optional[Mapping[str, object]] = None,
    entry_function_name: Optional[str] = None,
) -> Set[str]:
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _collect_load_derived_from_function(
            tree,
            provided_args=provided_args,
        )

    if isinstance(tree, ast.Module):
        if entry_function_name:
            entry_func = _find_function_by_name(tree, entry_function_name)
            if isinstance(entry_func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return _collect_load_derived_from_function(
                    entry_func,
                    provided_args=provided_args,
                )
        return _legacy_collect_load_derived_symbols(tree)

    return set()
