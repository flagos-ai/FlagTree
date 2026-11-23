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
from typing import Dict, Mapping, Optional

from .role_engine import TensorRole, infer_value_states
from .symbolic_expr import SymbolicExpr


def _is_slice_all(node: ast.AST) -> bool:
    return (isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None)


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


def _extract_axis_symbol_from_subscript_base(base_node: ast.AST) -> Optional[str]:
    if isinstance(base_node, ast.Name):
        return base_node.id
    if _is_tl_arange_call(base_node) and isinstance(base_node, ast.Call):
        stop_expr = _get_arange_stop_expr(base_node)
        if stop_expr is not None:
            return str(SymbolicExpr.from_ast(stop_expr))

    arange_stop_symbols = set()
    for node in ast.walk(base_node):
        if not _is_tl_arange_call(node):
            continue
        if not isinstance(node, ast.Call):
            continue
        stop_expr = _get_arange_stop_expr(node)
        if stop_expr is None:
            continue
        arange_stop_symbols.add(str(SymbolicExpr.from_ast(stop_expr)))
    if len(arange_stop_symbols) == 1:
        return next(iter(arange_stop_symbols))
    return None


def _extract_subscript_axes(node: ast.Subscript) -> Optional[tuple]:
    symbol_name = _extract_axis_symbol_from_subscript_base(node.value)
    if symbol_name is None:
        return None
    slice_node = node.slice
    if not isinstance(slice_node, ast.Tuple):
        return None
    if len(slice_node.elts) != 2:
        return None
    first_dim = slice_node.elts[0]
    second_dim = slice_node.elts[1]
    # var[:, None] => row axis
    if _is_slice_all(first_dim) and isinstance(second_dim, ast.Constant) and second_dim.value is None:
        return symbol_name, 0
    # var[None, :] => col axis
    if isinstance(first_dim, ast.Constant) and first_dim.value is None and _is_slice_all(second_dim):
        return symbol_name, 1
    return None


def infer_tensor_role_from_expr(expr: ast.AST) -> TensorRole:
    row_symbol = None
    col_symbol = None
    for node in ast.walk(expr):
        if not isinstance(node, ast.Subscript):
            continue
        axes = _extract_subscript_axes(node)
        if axes is None:
            continue
        symbol_name, axis_idx = axes
        if axis_idx == 0 and row_symbol is None:
            row_symbol = SymbolicExpr(text=symbol_name)
        if axis_idx == 1 and col_symbol is None:
            col_symbol = SymbolicExpr(text=symbol_name)
    return TensorRole(
        row_symbol=row_symbol,
        col_symbol=col_symbol,
        total_row_symbol=row_symbol,
        total_col_symbol=col_symbol,
    )


def annotate_tensor_roles(
    func_ast: ast.AST,
    inherited_roles: Optional[Mapping[str, TensorRole]] = None,
) -> Dict[str, TensorRole]:
    states = infer_value_states(func_ast, inherited_roles=inherited_roles)
    roles: Dict[str, TensorRole] = {}
    for name, state in states.items():
        if state.kind != "tensor" or state.role is None:
            continue
        if state.role.row_symbol is None and state.role.col_symbol is None:
            continue
        roles[name] = state.role

    # Keep legacy fallback from explicit subscript patterns to avoid regressions
    # in kernels that are not yet covered by role_engine rules.
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in roles:
            continue
        role = infer_tensor_role_from_expr(node.value)
        if role.row_symbol is None or role.col_symbol is None:
            continue
        roles[target.id] = role
    return roles
