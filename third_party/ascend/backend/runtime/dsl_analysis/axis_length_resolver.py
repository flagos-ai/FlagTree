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

from .load_semantics import (
    build_assignment_expr_map,
    collect_axis_candidates,
    extract_axis_block_symbol_map,
)
from .schema import (AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE,
                     AXIS_LENGTH_STATE_TUNABLE, AxisLengthStateInfo, SignatureInfo)
from .symbolic_expr import SymbolicExpr


def _is_tl_arange_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "arange")


def _get_arange_stop_expr(call_node: ast.Call) -> Optional[ast.AST]:
    if len(call_node.args) >= 2:
        return call_node.args[1]
    if len(call_node.args) == 1:
        return call_node.args[0]
    for keyword in call_node.keywords:
        if keyword.arg in ("end", "stop"):
            return keyword.value
    return None


def _is_static_number(value: object) -> bool:
    # bool is a subclass of int in Python; keep it explicit for readability.
    return isinstance(value, (bool, int, float))


def _evaluate_static_expr(node: ast.AST, provided_args: Mapping[str, object]) -> Optional[object]:
    if isinstance(node, ast.Constant):
        value = node.value
        if _is_static_number(value):
            return value
        return None

    if isinstance(node, ast.Name):
        value = provided_args.get(node.id, None)
        if _is_static_number(value):
            return value
        return None

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_static_expr(node.operand, provided_args)
        if operand is None:
            return None
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.Not):
            return not operand
        return None

    if isinstance(node, ast.BinOp):
        lhs = _evaluate_static_expr(node.left, provided_args)
        rhs = _evaluate_static_expr(node.right, provided_args)
        if lhs is None or rhs is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, ast.Div):
                return lhs / rhs
            if isinstance(node.op, ast.FloorDiv):
                return lhs // rhs
            if isinstance(node.op, ast.Mod):
                return lhs % rhs
            if isinstance(node.op, ast.Pow):
                return lhs**rhs
        except Exception:
            return None
        return None

    if isinstance(node, ast.BoolOp):
        values = []
        for item in node.values:
            evaluated = _evaluate_static_expr(item, provided_args)
            if evaluated is None:
                return None
            values.append(bool(evaluated))
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        return None

    if isinstance(node, ast.Compare):
        left = _evaluate_static_expr(node.left, provided_args)
        if left is None:
            return None
        comparisons = []
        cur_left = left
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate_static_expr(comparator, provided_args)
            if right is None:
                return None
            if isinstance(op, ast.Lt):
                comparisons.append(cur_left < right)
            elif isinstance(op, ast.LtE):
                comparisons.append(cur_left <= right)
            elif isinstance(op, ast.Gt):
                comparisons.append(cur_left > right)
            elif isinstance(op, ast.GtE):
                comparisons.append(cur_left >= right)
            elif isinstance(op, ast.Eq):
                comparisons.append(cur_left == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(cur_left != right)
            else:
                return None
            cur_left = right
        return all(comparisons)

    return None


def _resolve_conditional_expr(node: ast.AST, provided_args: Mapping[str, object]) -> ast.AST:
    resolved = node
    while isinstance(resolved, ast.IfExp):
        cond_value = _evaluate_static_expr(resolved.test, provided_args)
        if not isinstance(cond_value, bool):
            break
        resolved = resolved.body if cond_value else resolved.orelse
    return resolved


def _normalize_arange_stop_symbol(stop_expr: ast.AST, provided_args: Mapping[str, object]) -> str:
    resolved_expr = _resolve_conditional_expr(stop_expr, provided_args)
    return str(SymbolicExpr.from_ast(resolved_expr))


def _extract_axis_length_symbol_map(
    func_ast: ast.AST,
    provided_args: Mapping[str, object],
) -> Dict[str, str]:
    assignment_expr_map = build_assignment_expr_map(
        func_ast,
        provided_args=provided_args,
    )
    axis_candidates = collect_axis_candidates(func_ast)
    raw_symbol_map = extract_axis_block_symbol_map(
        func_ast,
        axis_candidates,
        assignment_expr_map=assignment_expr_map,
    )

    symbol_map = {}
    for symbol, raw_value in raw_symbol_map.items():
        try:
            raw_node = ast.parse(raw_value, mode="eval").body
        except Exception:
            symbol_map[symbol] = raw_value
            continue
        symbol_map[symbol] = _normalize_arange_stop_symbol(raw_node, provided_args)
    return symbol_map


def _parse_literal_int(text: str) -> Optional[int]:
    try:
        node = ast.parse(text, mode="eval").body
    except Exception:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _extract_expr_names(text: str) -> Set[str]:
    try:
        expr = ast.parse(text, mode="eval")
    except Exception:
        return set()
    names = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _classify_length_symbol(
    length_symbol: Optional[str],
    signature: SignatureInfo,
    provided_args: Mapping[str, object],
) -> Tuple[str, Optional[int]]:
    if length_symbol is None:
        return AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE, None

    literal_value = _parse_literal_int(length_symbol)
    if literal_value is not None:
        return AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, literal_value

    if signature.is_constexpr(length_symbol):
        if length_symbol in provided_args:
            value = provided_args[length_symbol]
            if isinstance(value, int):
                return AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, int(value)
            return AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, None
        return AXIS_LENGTH_STATE_TUNABLE, None

    expr_names = _extract_expr_names(length_symbol)
    if not expr_names:
        return AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE, None

    all_constexpr = all(signature.is_constexpr(name) for name in expr_names)
    if all_constexpr:
        if all(name in provided_args for name in expr_names):
            return AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, None
        return AXIS_LENGTH_STATE_TUNABLE, None
    return AXIS_LENGTH_STATE_RUNTIME_NON_TUNABLE, None


def classify_length_symbol(
    length_symbol: Optional[str],
    signature: SignatureInfo,
    provided_args: Mapping[str, object],
) -> Tuple[str, Optional[int]]:
    return _classify_length_symbol(length_symbol, signature, provided_args)


def resolve_axis_length_states(
    func_ast: ast.AST,
    axis_symbols: Iterable[str],
    signature: SignatureInfo,
    provided_args: Mapping[str, object],
) -> Dict[str, AxisLengthStateInfo]:
    axis_length_symbol_map = _extract_axis_length_symbol_map(
        func_ast,
        provided_args=provided_args,
    )
    resolved = {}
    seen = set()
    for axis_symbol in axis_symbols:
        if axis_symbol in seen:
            continue
        seen.add(axis_symbol)
        # When axis symbols are already direct length parameters
        # (e.g., BLOCK_M/BLOCK_N from block_ptr-based inference), classify them
        # directly instead of forcing a tl.arange indirection.
        length_symbol = axis_length_symbol_map.get(axis_symbol, axis_symbol)
        state, value = _classify_length_symbol(length_symbol, signature, provided_args)
        resolved[axis_symbol] = AxisLengthStateInfo(
            axis_symbol=axis_symbol,
            length_symbol=length_symbol,
            state=state,
            value=value,
        )
    return resolved
