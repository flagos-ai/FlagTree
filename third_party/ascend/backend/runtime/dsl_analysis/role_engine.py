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
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

from .load_semantics import (
    build_assignment_expr_map,
    collect_axis_candidates,
    extract_axis_total_symbol_map as extract_axis_total_symbol_map_impl,
    extract_name_depths_transitive,
    extract_name_ids,
    extract_name_ids_transitive,
    extract_param_names,
    pick_param_candidate,
)
from .symbolic_expr import SymbolicExpr


@dataclass(frozen=True)
class TensorRole:
    row_symbol: Optional[SymbolicExpr]
    col_symbol: Optional[SymbolicExpr]
    total_row_symbol: Optional[SymbolicExpr] = None
    total_col_symbol: Optional[SymbolicExpr] = None


@dataclass(frozen=True)
class ValueState:
    kind: str  # tensor/scalar/unknown
    role: Optional[TensorRole] = None


@dataclass(frozen=True)
class RuleContext:
    call_node: ast.Call


TransferFn = Callable[[RuleContext, List[ValueState]], ValueState]


class RoleTransferRegistry:

    def __init__(self):
        self._handlers: Dict[str, TransferFn] = {}

    def register(self, op_key: str, fn: TransferFn) -> None:
        self._handlers[op_key] = fn

    def get(self, op_key: str) -> Optional[TransferFn]:
        return self._handlers.get(op_key, None)


def _tensor(role: Optional[TensorRole]) -> ValueState:
    return ValueState(kind="tensor", role=role)


def _scalar() -> ValueState:
    return ValueState(kind="scalar", role=None)


def _unknown() -> ValueState:
    return ValueState(kind="unknown", role=None)


def _is_tensor(state: ValueState) -> bool:
    return state.kind == "tensor" and state.role is not None


def _merge_tensor_role(lhs: Optional[TensorRole], rhs: Optional[TensorRole]) -> Optional[TensorRole]:
    if lhs is not None and rhs is not None:
        if lhs == rhs:
            return lhs
        # Prefer preserving row from lhs and col from rhs when compatible.
        row = lhs.row_symbol if lhs.row_symbol is not None else rhs.row_symbol
        col = lhs.col_symbol if lhs.col_symbol is not None else rhs.col_symbol
        total_row = lhs.total_row_symbol if lhs.total_row_symbol is not None else rhs.total_row_symbol
        total_col = lhs.total_col_symbol if lhs.total_col_symbol is not None else rhs.total_col_symbol
        return TensorRole(
            row_symbol=row,
            col_symbol=col,
            total_row_symbol=total_row,
            total_col_symbol=total_col,
        )
    if lhs is not None:
        return lhs
    if rhs is not None:
        return rhs
    return None


def _swap_role(role: Optional[TensorRole]) -> Optional[TensorRole]:
    if role is None:
        return None
    return TensorRole(
        row_symbol=role.col_symbol,
        col_symbol=role.row_symbol,
        total_row_symbol=role.total_col_symbol,
        total_col_symbol=role.total_row_symbol,
    )


def _to_op_key(call_node: ast.Call) -> str:

    def _extract_tl_attr_tail(node: ast.AST) -> Optional[str]:
        attrs = []
        cur = node
        while isinstance(cur, ast.Attribute):
            attrs.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id == "tl" and attrs:
            return attrs[0]
        return None

    if isinstance(call_node.func, ast.Attribute):
        tl_attr = _extract_tl_attr_tail(call_node.func)
        if tl_attr is not None:
            return "tl.{}".format(tl_attr)
        return "method.{}".format(call_node.func.attr)
    if isinstance(call_node.func, ast.Name):
        return "call.{}".format(call_node.func.id)
    return "unknown"


def _is_tl_namespace_attr(node: ast.AST) -> bool:
    cur = node
    saw_attr = False
    while isinstance(cur, ast.Attribute):
        saw_attr = True
        cur = cur.value
    return saw_attr and isinstance(cur, ast.Name) and cur.id == "tl"


def _extract_make_block_ptr_role(call_node: ast.Call) -> Optional[TensorRole]:
    # Triton make_block_ptr supports both keyword and positional style:
    #   make_block_ptr(base, shape, strides, offsets, block_shape, order)
    shape_node = None
    block_shape_node = None
    keyword_map = {}
    for keyword in call_node.keywords:
        if keyword.arg is None:
            continue
        keyword_map[keyword.arg] = keyword.value
        if keyword.arg == "shape":
            shape_node = keyword.value
            continue
        if keyword.arg == "block_shape":
            block_shape_node = keyword.value
            continue
    if shape_node is None:
        shape_node = keyword_map.get("shape", None)
    if block_shape_node is None:
        block_shape_node = keyword_map.get("block_shape", None)
    if shape_node is None and len(call_node.args) >= 2:
        shape_node = call_node.args[1]
    if block_shape_node is None and len(call_node.args) >= 5:
        block_shape_node = call_node.args[4]
    if not isinstance(shape_node, ast.Tuple):
        return None
    if len(shape_node.elts) != 2:
        return None
    if not isinstance(block_shape_node, ast.Tuple):
        return None
    if len(block_shape_node.elts) != 2:
        return None
    return TensorRole(
        row_symbol=SymbolicExpr.from_ast(block_shape_node.elts[0]),
        col_symbol=SymbolicExpr.from_ast(block_shape_node.elts[1]),
        total_row_symbol=SymbolicExpr.from_ast(shape_node.elts[0]),
        total_col_symbol=SymbolicExpr.from_ast(shape_node.elts[1]),
    )


def _extract_shape_constructor_role(call_node: ast.Call) -> Optional[TensorRole]:
    shape_node = None
    for keyword in call_node.keywords:
        if keyword.arg == "shape":
            shape_node = keyword.value
            break
    if shape_node is None and call_node.args:
        shape_node = call_node.args[0]
    if not isinstance(shape_node, (ast.Tuple, ast.List)):
        return None
    if len(shape_node.elts) != 2:
        return None
    # Constructor shapes represent local tile extents. Keep total symbols empty
    # so later merge steps can upgrade to global axis totals from block pointers.
    return TensorRole(
        row_symbol=SymbolicExpr.from_ast(shape_node.elts[0]),
        col_symbol=SymbolicExpr.from_ast(shape_node.elts[1]),
        total_row_symbol=None,
        total_col_symbol=None,
    )


def _rule_make_block_ptr(ctx: RuleContext, _: List[ValueState]) -> ValueState:
    return _tensor(_extract_make_block_ptr_role(ctx.call_node))


def _rule_shape_constructor(ctx: RuleContext, _: List[ValueState]) -> ValueState:
    role = _extract_shape_constructor_role(ctx.call_node)
    if role is None:
        return _unknown()
    return _tensor(role)


def _rule_load(_: RuleContext, args: List[ValueState]) -> ValueState:
    if args and _is_tensor(args[0]):
        return args[0]
    return _unknown()


def _rule_transpose(_: RuleContext, args: List[ValueState]) -> ValueState:
    if not args or not _is_tensor(args[0]):
        return _unknown()
    return _tensor(_swap_role(args[0].role))


def _rule_where(_: RuleContext, args: List[ValueState]) -> ValueState:
    # tl.where(cond, a, b)
    if len(args) < 3:
        return _unknown()
    role = _merge_tensor_role(
        args[1].role if _is_tensor(args[1]) else None,
        args[2].role if _is_tensor(args[2]) else None,
    )
    if role is None:
        return _unknown()
    return _tensor(role)


def _rule_dot(_: RuleContext, args: List[ValueState]) -> ValueState:
    if len(args) < 2:
        return _unknown()
    lhs = args[0]
    rhs = args[1]
    if not _is_tensor(lhs) or not _is_tensor(rhs):
        return _unknown()
    lhs_role = lhs.role
    rhs_role = rhs.role
    if lhs_role is None or rhs_role is None:
        return _unknown()
    return _tensor(
        TensorRole(
            row_symbol=lhs_role.row_symbol,
            col_symbol=rhs_role.col_symbol,
            total_row_symbol=lhs_role.total_row_symbol or lhs_role.row_symbol,
            total_col_symbol=rhs_role.total_col_symbol or rhs_role.col_symbol,
        ))


def _rule_unary_passthrough(_: RuleContext, args: List[ValueState]) -> ValueState:
    if not args or not _is_tensor(args[0]):
        return _unknown()
    return args[0]


def _rule_binary_elementwise(_: RuleContext, args: List[ValueState]) -> ValueState:
    if len(args) < 2:
        return _unknown()
    role = _merge_tensor_role(
        args[0].role if _is_tensor(args[0]) else None,
        args[1].role if _is_tensor(args[1]) else None,
    )
    if role is None:
        return _unknown()
    return _tensor(role)


def _build_registry() -> RoleTransferRegistry:
    registry = RoleTransferRegistry()
    registry.register("tl.make_block_ptr", _rule_make_block_ptr)
    registry.register("tl.full", _rule_shape_constructor)
    registry.register("tl.zeros", _rule_shape_constructor)
    registry.register("tl.zeros_like", _rule_unary_passthrough)
    registry.register("tl.load", _rule_load)
    registry.register("tl.trans", _rule_transpose)
    registry.register("tl.transpose", _rule_transpose)
    registry.register("tl.where", _rule_where)
    registry.register("tl.dot", _rule_dot)
    registry.register("tl.conv", _rule_dot)
    registry.register("method.to", _rule_unary_passthrough)
    return registry


_REGISTRY = _build_registry()

_TL_UNARY_ELEMENTWISE = {
    "abs",
    "exp",
    "exp2",
    "log",
    "sqrt",
    "cos",
    "sin",
    "sigmoid",
}

_TL_BINARY_ELEMENTWISE = {
    "maximum",
    "minimum",
}


def _is_slice_all(node: ast.AST) -> bool:
    return (isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None)


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
    if _is_slice_all(first_dim) and isinstance(second_dim, ast.Constant) and second_dim.value is None:
        return symbol_name, 0
    if isinstance(first_dim, ast.Constant) and first_dim.value is None and _is_slice_all(second_dim):
        return symbol_name, 1
    return None


def _extract_subscript_axis_kind(node: ast.Subscript) -> Optional[int]:
    slice_node = node.slice
    if not isinstance(slice_node, ast.Tuple):
        return None
    if len(slice_node.elts) != 2:
        return None
    first_dim = slice_node.elts[0]
    second_dim = slice_node.elts[1]
    if _is_slice_all(first_dim) and isinstance(second_dim, ast.Constant) and second_dim.value is None:
        return 0
    if isinstance(first_dim, ast.Constant) and first_dim.value is None and _is_slice_all(second_dim):
        return 1
    return None


def _is_tl_arange_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == "arange")


def _extract_axis_symbols_from_expr(expr: ast.AST, axis_candidates: set) -> set:
    symbols = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in axis_candidates:
            symbols.add(node.id)
            continue
        if not isinstance(node, ast.Subscript):
            continue
        axes = _extract_subscript_axes(node)
        if axes is not None:
            symbol_name, _ = axes
            if symbol_name in axis_candidates:
                symbols.add(symbol_name)
            continue
        if isinstance(node.value, ast.Name) and node.value.id in axis_candidates:
            symbols.add(node.value.id)
    return symbols


def _infer_role_from_subscript_expr(
    expr: ast.AST,
    axis_total_symbol_map: Optional[Mapping[str, str]] = None,
) -> Optional[TensorRole]:
    row_symbol = None
    col_symbol = None
    total_row_symbol = None
    total_col_symbol = None
    total_symbol_map = axis_total_symbol_map or {}
    for node in ast.walk(expr):
        if not isinstance(node, ast.Subscript):
            continue
        axes = _extract_subscript_axes(node)
        if axes is None:
            continue
        symbol_name, axis_idx = axes
        total_name = total_symbol_map.get(symbol_name, symbol_name)
        if axis_idx == 0 and row_symbol is None:
            row_symbol = SymbolicExpr(text=symbol_name)
            total_row_symbol = SymbolicExpr(text=total_name)
        if axis_idx == 1 and col_symbol is None:
            col_symbol = SymbolicExpr(text=symbol_name)
            total_col_symbol = SymbolicExpr(text=total_name)
    # Keep partial-axis role so binary expression merging can reconstruct
    # full row/col roles through pointer-hoisted forms.
    if row_symbol is None and col_symbol is None:
        return None
    return TensorRole(
        row_symbol=row_symbol,
        col_symbol=col_symbol,
        total_row_symbol=total_row_symbol,
        total_col_symbol=total_col_symbol,
    )


def _infer_axis_alias_role_for_assign(
    target_name: str,
    expr: ast.AST,
    axis_total_symbol_map: Optional[Mapping[str, str]] = None,
) -> Optional[TensorRole]:
    row_symbol = None
    col_symbol = None
    total_row_symbol = None
    total_col_symbol = None
    total_name = (axis_total_symbol_map or {}).get(target_name, target_name)
    for node in ast.walk(expr):
        if not isinstance(node, ast.Subscript):
            continue
        axis_kind = _extract_subscript_axis_kind(node)
        if axis_kind is None:
            continue
        if axis_kind == 0 and row_symbol is None:
            row_symbol = SymbolicExpr(text=target_name)
            total_row_symbol = SymbolicExpr(text=total_name)
        if axis_kind == 1 and col_symbol is None:
            col_symbol = SymbolicExpr(text=target_name)
            total_col_symbol = SymbolicExpr(text=total_name)

    if row_symbol is None and col_symbol is None:
        return None

    return TensorRole(
        row_symbol=row_symbol,
        col_symbol=col_symbol,
        total_row_symbol=total_row_symbol,
        total_col_symbol=total_col_symbol,
    )


def _extract_axis_total_symbol_map(func_ast: ast.AST) -> Dict[str, str]:
    param_names = extract_param_names(func_ast)
    assignment_expr_map = build_assignment_expr_map(func_ast)
    axis_candidates = collect_axis_candidates(func_ast, param_names=param_names)
    return extract_axis_total_symbol_map_impl(
        func_ast,
        assignment_expr_map=assignment_expr_map,
        axis_candidates=axis_candidates,
        param_names=param_names,
    )


def _infer_expr_state(
    expr: ast.AST,
    env: Mapping[str, ValueState],
    axis_total_symbol_map: Optional[Mapping[str, str]] = None,
) -> ValueState:
    if isinstance(expr, ast.Name):
        return env.get(expr.id, _unknown())

    if isinstance(expr, ast.Constant):
        return _scalar()

    if isinstance(expr, ast.Subscript):
        if isinstance(expr.value, ast.Name):
            base = env.get(expr.value.id, _unknown())
            if _is_tensor(base):
                return base
        role = _infer_role_from_subscript_expr(
            expr,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        if role is not None:
            return _tensor(role)
        return _unknown()

    if isinstance(expr, ast.UnaryOp):
        operand = _infer_expr_state(
            expr.operand,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        if _is_tensor(operand):
            return operand
        return _unknown()

    if isinstance(expr, ast.BinOp):
        lhs = _infer_expr_state(
            expr.left,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        rhs = _infer_expr_state(
            expr.right,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        return _rule_binary_elementwise(
            RuleContext(call_node=ast.Call(func=ast.Name(id="binop"), args=[], keywords=[])),
            [lhs, rhs],
        )

    if isinstance(expr, ast.IfExp):
        body_state = _infer_expr_state(
            expr.body,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        else_state = _infer_expr_state(
            expr.orelse,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        return _rule_binary_elementwise(
            RuleContext(call_node=ast.Call(func=ast.Name(id="ifexp"), args=[], keywords=[])),
            [body_state, else_state],
        )

    if isinstance(expr, ast.Attribute):
        base_state = _infer_expr_state(
            expr.value,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        if expr.attr == "T" and _is_tensor(base_state):
            return _tensor(_swap_role(base_state.role))
        if expr.attr == "dtype":
            return _scalar()
        return _unknown()

    if isinstance(expr, ast.Call):
        op_key = _to_op_key(expr)
        arg_states: List[ValueState] = []
        if (isinstance(expr.func, ast.Attribute) and not _is_tl_namespace_attr(expr.func)):
            # Method calls (e.g. tensor.to(...)) should propagate from receiver.
            arg_states.append(_infer_expr_state(
                expr.func.value,
                env,
                axis_total_symbol_map=axis_total_symbol_map,
            ))
        arg_states.extend(
            _infer_expr_state(
                arg,
                env,
                axis_total_symbol_map=axis_total_symbol_map,
            ) for arg in expr.args)

        handler = _REGISTRY.get(op_key)
        if handler is not None:
            state = handler(RuleContext(call_node=expr), arg_states)
            if state.kind != "unknown":
                return state

        # Fallback by operator family.
        if op_key.startswith("tl."):
            op_name = op_key[len("tl."):]
            if op_name in _TL_UNARY_ELEMENTWISE:
                return _rule_unary_passthrough(RuleContext(call_node=expr), arg_states)
            if op_name in _TL_BINARY_ELEMENTWISE:
                return _rule_binary_elementwise(RuleContext(call_node=expr), arg_states)

        if arg_states and _is_tensor(arg_states[0]):
            return arg_states[0]
        return _unknown()

    return _unknown()


def _assign_target(target: ast.AST, state: ValueState, env: Dict[str, ValueState]) -> None:
    if not isinstance(target, ast.Name):
        return
    env[target.id] = state


def _merge_env(dst: Dict[str, ValueState], src: Mapping[str, ValueState]) -> None:
    for name, state in src.items():
        if name not in dst:
            dst[name] = state
            continue
        cur = dst[name]
        if cur == state:
            continue
        if _is_tensor(cur) and _is_tensor(state):
            merged = _merge_tensor_role(cur.role, state.role)
            dst[name] = _tensor(merged)
            continue
        if _is_tensor(cur):
            continue
        if _is_tensor(state):
            dst[name] = state


def _is_single_axis_role(role: Optional[TensorRole]) -> bool:
    if role is None:
        return False
    has_row = role.row_symbol is not None
    has_col = role.col_symbol is not None
    return has_row != has_col


def _visit_stmt(
    stmt: ast.stmt,
    env: Dict[str, ValueState],
    axis_total_symbol_map: Optional[Mapping[str, str]] = None,
) -> None:
    if isinstance(stmt, ast.Assign):
        state = _infer_expr_state(
            stmt.value,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        # Broadcasted axis aliases like:
        #   offs_m = ... + tl.arange(... )[:, None]
        #   offs_k = tl.arange(... )[None, :]
        # should keep alias symbols (offs_m/offs_k) for downstream axis tracking.
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            alias_role = _infer_axis_alias_role_for_assign(
                stmt.targets[0].id,
                stmt.value,
                axis_total_symbol_map=axis_total_symbol_map,
            )
            if alias_role is not None and (not _is_tensor(state) or
                                           (_is_single_axis_role(alias_role) and _is_single_axis_role(state.role))):
                state = _tensor(alias_role)
        for target in stmt.targets:
            _assign_target(target, state, env)
        return

    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return
        state = _infer_expr_state(
            stmt.value,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        _assign_target(stmt.target, state, env)
        return

    if isinstance(stmt, ast.AugAssign):
        target_state = _infer_expr_state(
            stmt.target,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        value_state = _infer_expr_state(
            stmt.value,
            env,
            axis_total_symbol_map=axis_total_symbol_map,
        )
        state = _rule_binary_elementwise(
            RuleContext(call_node=ast.Call(func=ast.Name(id="augassign"), args=[], keywords=[])),
            [target_state, value_state],
        )
        if state.kind == "unknown":
            if target_state.kind != "unknown":
                state = target_state
            elif value_state.kind != "unknown":
                state = value_state
        _assign_target(stmt.target, state, env)
        return

    if isinstance(stmt, ast.For):
        inner_env = dict(env)
        for child in stmt.body:
            _visit_stmt(child, inner_env, axis_total_symbol_map=axis_total_symbol_map)
        _merge_env(env, inner_env)
        orelse_env = dict(env)
        for child in stmt.orelse:
            _visit_stmt(child, orelse_env, axis_total_symbol_map=axis_total_symbol_map)
        _merge_env(env, orelse_env)
        return

    if isinstance(stmt, ast.While):
        inner_env = dict(env)
        for child in stmt.body:
            _visit_stmt(child, inner_env, axis_total_symbol_map=axis_total_symbol_map)
        _merge_env(env, inner_env)
        return

    if isinstance(stmt, ast.If):
        body_env = dict(env)
        for child in stmt.body:
            _visit_stmt(child, body_env, axis_total_symbol_map=axis_total_symbol_map)
        else_env = dict(env)
        for child in stmt.orelse:
            _visit_stmt(child, else_env, axis_total_symbol_map=axis_total_symbol_map)
        _merge_env(env, body_env)
        _merge_env(env, else_env)
        return

    if isinstance(stmt, ast.With):
        inner_env = dict(env)
        for child in stmt.body:
            _visit_stmt(child, inner_env, axis_total_symbol_map=axis_total_symbol_map)
        _merge_env(env, inner_env)
        return


def infer_value_states(
    func_ast: ast.AST,
    inherited_roles: Optional[Mapping[str, TensorRole]] = None,
) -> Dict[str, ValueState]:
    axis_total_symbol_map = _extract_axis_total_symbol_map(func_ast)
    env: Dict[str, ValueState] = {name: _tensor(role) for name, role in (inherited_roles or {}).items()}

    if isinstance(func_ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = func_ast.body
    else:
        body = [node for node in ast.walk(func_ast) if isinstance(node, ast.stmt)]

    for stmt in body:
        _visit_stmt(stmt, env, axis_total_symbol_map=axis_total_symbol_map)
    return env
