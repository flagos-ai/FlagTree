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
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set

from .call_graph import build_call_sites, build_function_index
from .dot_role_annotator import TensorRole, annotate_tensor_roles
from .symbolic_expr import SymbolicExpr


def _is_tl_mix_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"
            and node.func.attr in ("dot", "conv"))


def _is_tl_transpose_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"
            and node.func.attr in ("trans", "transpose"))


def _merge_tensor_name_candidates(*candidates: Optional[str]) -> Optional[str]:
    names = []
    for candidate in candidates:
        if candidate is None or candidate in names:
            continue
        names.append(candidate)
    if len(names) == 1:
        return names[0]
    return None


def _extract_tensor_name_from_expr(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id

    if isinstance(expr, ast.Subscript):
        if isinstance(expr.value, ast.Name):
            return expr.value.id
        return _extract_tensor_name_from_expr(expr.value)

    if isinstance(expr, ast.Call):
        if isinstance(expr.func, ast.Attribute):
            if isinstance(expr.func.value, ast.Name):
                # tl.xxx(tensor) wrappers should resolve from positional args,
                # while tensor.xxx(...) methods can use receiver name directly.
                if expr.func.value.id != "tl":
                    return expr.func.value.id
            elif not (isinstance(expr.func.value, ast.Name) and expr.func.value.id == "tl"):
                base = _extract_tensor_name_from_expr(expr.func.value)
                if base is not None:
                    return base
        if expr.args:
            return _extract_tensor_name_from_expr(expr.args[0])
        return None

    if isinstance(expr, ast.UnaryOp):
        return _extract_tensor_name_from_expr(expr.operand)

    if isinstance(expr, ast.BinOp):
        return _merge_tensor_name_candidates(
            _extract_tensor_name_from_expr(expr.left),
            _extract_tensor_name_from_expr(expr.right),
        )

    if isinstance(expr, ast.IfExp):
        return _merge_tensor_name_candidates(
            _extract_tensor_name_from_expr(expr.body),
            _extract_tensor_name_from_expr(expr.orelse),
        )

    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        return expr.value.id

    return None


@dataclass(frozen=True)
class DotSiteMNK:
    site_id: int
    m: Optional[str]
    n: Optional[str]
    k: Optional[str]
    lhs: Optional[str]
    rhs: Optional[str]
    m_total: Optional[str] = None
    n_total: Optional[str] = None
    k_total: Optional[str] = None


def _dedup_key(site: DotSiteMNK):
    return (
        site.lhs,
        site.rhs,
        site.m,
        site.n,
        site.k,
        site.m_total,
        site.n_total,
        site.k_total,
    )


def _deduplicate_dot_sites(dot_sites: List[DotSiteMNK]) -> List[DotSiteMNK]:
    seen = set()
    deduped: List[DotSiteMNK] = []
    for site in dot_sites:
        key = _dedup_key(site)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(site)
    return [replace(site, site_id=idx) for idx, site in enumerate(deduped)]


class _DotSiteCollector(ast.NodeVisitor):

    def __init__(self, tensor_roles):
        self.tensor_roles = tensor_roles
        self.dot_sites = []
        self._site_id = 0

    @staticmethod
    def _swap_role(role: Optional[TensorRole]) -> Optional[TensorRole]:
        if role is None:
            return None
        return TensorRole(
            row_symbol=role.col_symbol,
            col_symbol=role.row_symbol,
            total_row_symbol=role.total_col_symbol,
            total_col_symbol=role.total_row_symbol,
        )

    def _resolve_tensor_name_and_role(self, expr: ast.AST):
        transposed = _is_tl_transpose_call(expr)
        base_expr = expr.args[0] if transposed and isinstance(expr, ast.Call) and expr.args else expr
        tensor_name, role = self._extract_tensor_name_and_role(base_expr)
        if transposed:
            role = self._swap_role(role)
        return tensor_name, role

    def _extract_tensor_name_and_role(self, expr: ast.AST):
        if isinstance(expr, ast.BinOp):
            left_name, left_role = self._extract_tensor_name_and_role(expr.left)
            right_name, right_role = self._extract_tensor_name_and_role(expr.right)
            if left_role is not None and right_role is None:
                return left_name, left_role
            if right_role is not None and left_role is None:
                return right_name, right_role
            if left_role is not None and right_role is not None and left_name == right_name:
                return left_name, left_role
            return None, None

        if isinstance(expr, ast.UnaryOp):
            return self._extract_tensor_name_and_role(expr.operand)

        if isinstance(expr, ast.IfExp):
            body_name, body_role = self._extract_tensor_name_and_role(expr.body)
            else_name, else_role = self._extract_tensor_name_and_role(expr.orelse)
            if body_role is not None and else_role is None:
                return body_name, body_role
            if else_role is not None and body_role is None:
                return else_name, else_role
            if body_role is not None and else_role is not None and body_name == else_name:
                return body_name, body_role
            return None, None

        tensor_name = self._extract_tensor_name(expr)
        if tensor_name is None:
            return None, None
        return tensor_name, self.tensor_roles.get(tensor_name, None)

    def _extract_tensor_name(self, expr: ast.AST) -> Optional[str]:
        return _extract_tensor_name_from_expr(expr)

    def visit_Call(self, node: ast.Call):
        if _is_tl_mix_call(node):
            lhs = None
            rhs = None
            lhs_role = None
            rhs_role = None
            if len(node.args) >= 2:
                lhs, lhs_role = self._resolve_tensor_name_and_role(node.args[0])
                rhs, rhs_role = self._resolve_tensor_name_and_role(node.args[1])

            m = str(lhs_role.row_symbol) if lhs_role and lhs_role.row_symbol else None
            n = str(rhs_role.col_symbol) if rhs_role and rhs_role.col_symbol else None
            k_lhs = str(lhs_role.col_symbol) if lhs_role and lhs_role.col_symbol else None
            k_rhs = str(rhs_role.row_symbol) if rhs_role and rhs_role.row_symbol else None
            k = k_lhs
            if k is None:
                k = k_rhs
            elif k_rhs is not None and k != k_rhs:
                k = "{}|{}".format(k_lhs, k_rhs)

            m_total = str(lhs_role.total_row_symbol) if lhs_role and lhs_role.total_row_symbol else None
            n_total = str(rhs_role.total_col_symbol) if rhs_role and rhs_role.total_col_symbol else None
            k_lhs_total = (str(lhs_role.total_col_symbol) if lhs_role and lhs_role.total_col_symbol else None)
            k_rhs_total = (str(rhs_role.total_row_symbol) if rhs_role and rhs_role.total_row_symbol else None)
            k_total = k_lhs_total
            if k_total is None:
                k_total = k_rhs_total
            elif k_rhs_total is not None and k_total != k_rhs_total:
                k_total = "{}|{}".format(k_lhs_total, k_rhs_total)

            self.dot_sites.append(
                DotSiteMNK(
                    site_id=self._site_id,
                    m=m,
                    n=n,
                    k=k,
                    lhs=lhs,
                    rhs=rhs,
                    m_total=m_total,
                    n_total=n_total,
                    k_total=k_total,
                ))
            self._site_id += 1

        self.generic_visit(node)


def infer_dot_sites_mnk(
    func_ast: ast.AST,
    inherited_roles: Optional[Dict[str, TensorRole]] = None,
) -> List[DotSiteMNK]:
    tensor_roles = annotate_tensor_roles(func_ast, inherited_roles=inherited_roles)
    collector = _DotSiteCollector(tensor_roles)
    collector.visit(func_ast)
    return _deduplicate_dot_sites(collector.dot_sites)


def _substitute_symbol(symbol: Optional[str], symbol_map: Dict[str, str]) -> Optional[str]:
    if symbol is None:
        return None
    return symbol_map.get(symbol, symbol)


def _expr_to_text(arg_node: ast.AST, symbol_map: Dict[str, str]) -> Optional[str]:
    if _is_tl_transpose_call(arg_node) and arg_node.args:
        arg_text = _expr_to_text(arg_node.args[0], symbol_map)
        if arg_text is None:
            return None
        return "tl.trans({})".format(arg_text)
    if isinstance(arg_node, ast.Name):
        return symbol_map.get(arg_node.id, arg_node.id)
    try:
        text = ast.unparse(arg_node)
    except Exception:
        return None
    return symbol_map.get(text, text)


def _substitute_role(
    role: Optional[TensorRole],
    symbol_map: Dict[str, str],
) -> Optional[TensorRole]:
    if role is None:
        return None

    def _map_symbol(sym: Optional[SymbolicExpr]) -> Optional[SymbolicExpr]:
        if sym is None:
            return None
        text = symbol_map.get(str(sym), str(sym))
        return SymbolicExpr(text=text)

    return TensorRole(
        row_symbol=_map_symbol(role.row_symbol),
        col_symbol=_map_symbol(role.col_symbol),
        total_row_symbol=_map_symbol(role.total_row_symbol),
        total_col_symbol=_map_symbol(role.total_col_symbol),
    )


def _resolve_role_for_expr(
    arg_node: ast.AST,
    tensor_roles: Dict[str, TensorRole],
) -> Optional[TensorRole]:
    transposed = _is_tl_transpose_call(arg_node)
    base_node = arg_node.args[0] if transposed and isinstance(arg_node, ast.Call) and arg_node.args else arg_node
    tensor_name = _extract_tensor_name_from_expr(base_node)
    if tensor_name is None:
        return None
    role = tensor_roles.get(tensor_name, None)
    if role is None:
        return None
    if not transposed:
        return role
    return TensorRole(
        row_symbol=role.col_symbol,
        col_symbol=role.row_symbol,
        total_row_symbol=role.total_col_symbol,
        total_col_symbol=role.total_row_symbol,
    )


def infer_dot_sites_mnk_interprocedural(module_ast: ast.AST, entry_function_name: str) -> List[DotSiteMNK]:
    functions = build_function_index(module_ast)
    if entry_function_name not in functions:
        return []

    call_sites = build_call_sites(functions)
    call_sites_by_caller = {}
    for site in call_sites:
        call_sites_by_caller.setdefault(site.caller, []).append(site)

    emitted = []
    next_site_id = 0

    def walk(
        function_name: str,
        symbol_map: Dict[str, str],
        param_roles: Dict[str, TensorRole],
        active: Set[str],
    ):
        nonlocal next_site_id
        if function_name in active:
            return
        active.add(function_name)

        function_meta = functions.get(function_name, None)
        if function_meta is None:
            active.remove(function_name)
            return

        local_tensor_roles = annotate_tensor_roles(
            function_meta.node,
            inherited_roles=param_roles,
        )
        local_dot_sites = infer_dot_sites_mnk(
            function_meta.node,
            inherited_roles=param_roles,
        )
        for site in local_dot_sites:
            emitted.append(
                DotSiteMNK(
                    site_id=next_site_id,
                    m=_substitute_symbol(site.m, symbol_map),
                    n=_substitute_symbol(site.n, symbol_map),
                    k=_substitute_symbol(site.k, symbol_map),
                    lhs=_substitute_symbol(site.lhs, symbol_map),
                    rhs=_substitute_symbol(site.rhs, symbol_map),
                    m_total=_substitute_symbol(site.m_total, symbol_map),
                    n_total=_substitute_symbol(site.n_total, symbol_map),
                    k_total=_substitute_symbol(site.k_total, symbol_map),
                ))
            next_site_id += 1

        for call_site in call_sites_by_caller.get(function_name, []):
            callee_meta = functions.get(call_site.callee, None)
            if callee_meta is None:
                continue
            callee_symbol_map = {}
            callee_param_roles: Dict[str, TensorRole] = {}
            for idx, param_name in enumerate(callee_meta.param_names):
                if idx >= len(call_site.args):
                    break
                arg_node = call_site.args[idx]
                if arg_node is None:
                    continue
                arg_text = _expr_to_text(arg_node, symbol_map)
                if arg_text is None:
                    continue
                callee_symbol_map[param_name] = arg_text
                arg_role = _resolve_role_for_expr(arg_node, local_tensor_roles)
                arg_role = _substitute_role(arg_role, symbol_map)
                if arg_role is not None:
                    callee_param_roles[param_name] = arg_role
            walk(call_site.callee, callee_symbol_map, callee_param_roles, active)

        active.remove(function_name)

    walk(entry_function_name, {}, {}, set())
    return _deduplicate_dot_sites(emitted)
