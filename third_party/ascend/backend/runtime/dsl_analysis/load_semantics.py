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
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


def ast_to_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ast.unparse(node)


def is_tl_call(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == name)


def iter_walk_in_order(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from iter_walk_in_order(child)


def get_arange_stop_expr(call_node: ast.Call) -> Optional[ast.AST]:
    if len(call_node.args) >= 2:
        return call_node.args[1]
    if len(call_node.args) == 1:
        return call_node.args[0]
    for keyword in call_node.keywords:
        if keyword.arg in ("end", "stop"):
            return keyword.value
    return None


def collect_arange_stop_texts(expr: ast.AST) -> List[str]:
    values: List[str] = []
    seen = set()
    for node in iter_walk_in_order(expr):
        if not is_tl_call(node, "arange"):
            continue
        stop_expr = get_arange_stop_expr(node)
        if stop_expr is None:
            continue
        text = ast_to_text(stop_expr)
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def is_slice_all(node: ast.AST) -> bool:
    return (isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None)


def is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def extract_axis_symbol_from_subscript_base(base_node: ast.AST) -> Optional[str]:
    if isinstance(base_node, ast.Name):
        return base_node.id
    if is_tl_call(base_node, "arange") and isinstance(base_node, ast.Call):
        stop_expr = get_arange_stop_expr(base_node)
        if stop_expr is not None:
            return ast_to_text(stop_expr)
    arange_lengths = collect_arange_stop_texts(base_node)
    if len(arange_lengths) == 1:
        return arange_lengths[0]
    return None


def extract_subscript_axis_info(node: ast.Subscript) -> Optional[Tuple[int, str, int]]:
    symbol = extract_axis_symbol_from_subscript_base(node.value)
    if symbol is None:
        return None
    if not isinstance(node.slice, ast.Tuple):
        return None
    dims = node.slice.elts
    if len(dims) < 2 or len(dims) > 4:
        return None
    axis_index = None
    for idx, dim in enumerate(dims):
        if is_slice_all(dim):
            if axis_index is not None:
                return None
            axis_index = idx
            continue
        if is_none_const(dim):
            continue
        return None
    if axis_index is None:
        return None
    return axis_index, symbol, len(dims)


def extract_target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for item in target.elts:
            names.extend(extract_target_names(item))
        return names
    return []


def to_static_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def resolve_static_scalar(expr: ast.AST, known_values: Mapping[str, object]) -> Optional[object]:
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
        operand = resolve_static_scalar(expr.operand, known_values)
        if operand is None:
            return None
        if isinstance(expr.op, ast.Not):
            operand_bool = to_static_bool(operand)
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
            for child in expr.values:
                value = resolve_static_bool_expr(child, known_values)
                if value is False:
                    return False
                if value is None:
                    maybe_unknown = True
            return None if maybe_unknown else True
        if isinstance(expr.op, ast.Or):
            maybe_unknown = False
            for child in expr.values:
                value = resolve_static_bool_expr(child, known_values)
                if value is True:
                    return True
                if value is None:
                    maybe_unknown = True
            return None if maybe_unknown else False
        return None
    if isinstance(expr, ast.Compare):
        if len(expr.ops) != 1 or len(expr.comparators) != 1:
            return None
        lhs = resolve_static_scalar(expr.left, known_values)
        rhs = resolve_static_scalar(expr.comparators[0], known_values)
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


def resolve_static_bool_expr(expr: ast.AST, known_values: Mapping[str, object]) -> Optional[bool]:
    scalar = resolve_static_scalar(expr, known_values)
    if scalar is None:
        return None
    return to_static_bool(scalar)


def clone_assignment_expr_map(src: Mapping[str, Sequence[ast.AST]]) -> Dict[str, List[ast.AST]]:
    return {name: list(exprs) for name, exprs in src.items()}


def append_unique_expr(values: List[ast.AST], expr: ast.AST) -> None:
    expr_dump = ast.dump(expr)
    for current in values:
        if ast.dump(current) == expr_dump:
            return
    values.append(expr)


def merge_assignment_expr_map(
    dst: Dict[str, List[ast.AST]],
    src: Mapping[str, Sequence[ast.AST]],
) -> None:
    for name, exprs in src.items():
        bucket = dst.setdefault(name, [])
        for expr in exprs:
            append_unique_expr(bucket, expr)


def set_assignment_expr(
    assignment_expr_map: Dict[str, List[ast.AST]],
    target_names: Sequence[str],
    value_expr: ast.AST,
) -> None:
    for name in target_names:
        assignment_expr_map[name] = [value_expr]


def visit_stmt_for_assignment_map(
    stmt: ast.stmt,
    assignment_expr_map: Dict[str, List[ast.AST]],
    known_values: Mapping[str, object],
) -> None:
    if isinstance(stmt, ast.Assign):
        target_names: List[str] = []
        for target in stmt.targets:
            target_names.extend(extract_target_names(target))
        if target_names:
            set_assignment_expr(assignment_expr_map, target_names, stmt.value)
        return

    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return
        target_names = extract_target_names(stmt.target)
        if target_names:
            set_assignment_expr(assignment_expr_map, target_names, stmt.value)
        return

    if isinstance(stmt, ast.If):
        cond = resolve_static_bool_expr(stmt.test, known_values)
        if cond is True:
            for child in stmt.body:
                visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
            return
        if cond is False:
            for child in stmt.orelse:
                visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
            return
        body_map = clone_assignment_expr_map(assignment_expr_map)
        else_map = clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            visit_stmt_for_assignment_map(child, body_map, known_values)
        for child in stmt.orelse:
            visit_stmt_for_assignment_map(child, else_map, known_values)
        assignment_expr_map.clear()
        merge_assignment_expr_map(assignment_expr_map, body_map)
        merge_assignment_expr_map(assignment_expr_map, else_map)
        return

    if isinstance(stmt, ast.For):
        loop_map = clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            visit_stmt_for_assignment_map(child, loop_map, known_values)
        for child in stmt.orelse:
            visit_stmt_for_assignment_map(child, loop_map, known_values)
        merge_assignment_expr_map(assignment_expr_map, loop_map)
        return

    if isinstance(stmt, ast.While):
        loop_map = clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            visit_stmt_for_assignment_map(child, loop_map, known_values)
        for child in stmt.orelse:
            visit_stmt_for_assignment_map(child, loop_map, known_values)
        merge_assignment_expr_map(assignment_expr_map, loop_map)
        return

    if isinstance(stmt, ast.With):
        for child in stmt.body:
            visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
        return


def build_assignment_expr_map(
    func_node: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
) -> Dict[str, List[ast.AST]]:
    known_values: Dict[str, object] = {}
    for name, value in (provided_args or {}).items():
        if isinstance(value, (bool, int, float)):
            known_values[name] = value

    result: Dict[str, List[ast.AST]] = {}
    body = getattr(func_node, "body", None)
    if not isinstance(body, list):
        return result
    for stmt in body:
        if not isinstance(stmt, ast.stmt):
            continue
        visit_stmt_for_assignment_map(stmt, result, known_values)
    return result


def extract_name_ids(expr: ast.AST) -> Set[str]:
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)}


def extract_name_depths_transitive(
    expr: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Dict[str, int]:
    name_depths = {name: 0 for name in extract_name_ids(expr)}
    queue = list(name_depths.keys())
    while queue:
        name = queue.pop()
        base_depth = name_depths[name]
        for assigned_expr in assignment_expr_map.get(name, ()):
            for sub_name in extract_name_ids(assigned_expr):
                next_depth = base_depth + 1
                cur_depth = name_depths.get(sub_name, None)
                if cur_depth is not None and cur_depth <= next_depth:
                    continue
                name_depths[sub_name] = next_depth
                queue.append(sub_name)
    return name_depths


def extract_name_ids_transitive(
    expr: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Set[str]:
    return set(extract_name_depths_transitive(expr, assignment_expr_map).keys())


def collect_symbol_dependency_names(
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Set[str]:
    names = {symbol}
    for expr in assignment_expr_map.get(symbol, ()):
        names.update(extract_name_ids_transitive(expr, assignment_expr_map))
    return names


def build_symbol_user_expr_map(assignment_expr_map: Mapping[str, Sequence[ast.AST]], ) -> Dict[str, List[ast.AST]]:
    result: Dict[str, List[ast.AST]] = {}
    for exprs in assignment_expr_map.values():
        for expr in exprs:
            seen = set()
            for node in ast.walk(expr):
                if not isinstance(node, ast.Subscript):
                    continue
                axis_info = extract_subscript_axis_info(node)
                if axis_info is None:
                    continue
                _, symbol, _ = axis_info
                if symbol in seen:
                    continue
                seen.add(symbol)
                result.setdefault(symbol, []).append(expr)
    return result


def pick_param_candidate(
    expr: ast.AST,
    *,
    param_names: Set[str],
    exclude: Set[str],
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Optional[str]:
    name_depths = extract_name_depths_transitive(expr, assignment_expr_map)
    candidates = [name for name, depth in name_depths.items() if name in param_names and name not in exclude]
    if not candidates:
        return None
    min_depth = min(name_depths[name] for name in candidates)
    candidates = sorted(name for name in candidates if name_depths[name] == min_depth)
    for preferred in ("M", "N", "K"):
        if preferred in candidates:
            return preferred
    return candidates[0]


def extract_axis_symbols_from_expr(expr: ast.AST, axis_candidates: Set[str]) -> Set[str]:
    symbols: Set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in axis_candidates:
            symbols.add(node.id)
            continue
        if not isinstance(node, ast.Subscript):
            continue
        axis_info = extract_subscript_axis_info(node)
        if axis_info is None:
            continue
        _, symbol, _ = axis_info
        if symbol in axis_candidates:
            symbols.add(symbol)
    return symbols


def extract_range_stop_and_step(node: ast.For) -> Tuple[Optional[ast.AST], Optional[ast.AST]]:
    if not isinstance(node.iter, ast.Call):
        return None, None
    iter_fn = node.iter.func
    is_range = isinstance(iter_fn, ast.Name) and iter_fn.id == "range"
    is_tl_range = (isinstance(iter_fn, ast.Attribute) and isinstance(iter_fn.value, ast.Name)
                   and iter_fn.value.id == "tl" and iter_fn.attr == "range")
    if not (is_range or is_tl_range):
        return None, None

    args = list(node.iter.args)
    if len(args) == 0:
        return None, None
    if len(args) == 1:
        return args[0], None
    if len(args) == 2:
        return args[1], None
    return args[1], args[2]


def extract_param_names(func_node: ast.AST) -> Set[str]:
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    args_node = func_node.args
    ordered_args = list(args_node.posonlyargs) + list(args_node.args) + list(args_node.kwonlyargs)
    return {arg.arg for arg in ordered_args if isinstance(arg, ast.arg)}


def collect_axis_candidates(func_node: ast.AST, param_names: Optional[Set[str]] = None) -> Set[str]:
    axis_candidates: Set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        if is_tl_call(node.value, "arange"):
            axis_candidates.add(target_name)
            continue
        arange_lengths = collect_arange_stop_texts(node.value)
        if len(arange_lengths) == 1:
            axis_candidates.add(target_name)
            continue
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mod):
            axis_candidates.add(target_name)
            continue
        for sub_node in ast.walk(node.value):
            if not isinstance(sub_node, ast.Subscript):
                continue
            axis_info = extract_subscript_axis_info(sub_node)
            if axis_info is None:
                continue
            axis_candidates.add(target_name)
            break

    if param_names:
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Subscript):
                continue
            axis_info = extract_subscript_axis_info(node)
            if axis_info is None:
                continue
            _, symbol, _ = axis_info
            if symbol in param_names:
                axis_candidates.add(symbol)
    return axis_candidates


def extract_axis_block_symbol_map(
    func_node: ast.AST,
    axis_candidates: Set[str],
    assignment_expr_map: Optional[Mapping[str, Sequence[ast.AST]]] = None,
) -> Dict[str, str]:
    axis_block_symbol_map: Dict[str, str] = {}
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        if target_name not in axis_candidates:
            continue
        arange_lengths = collect_arange_stop_texts(node.value)
        if len(arange_lengths) == 1:
            axis_block_symbol_map[target_name] = arange_lengths[0]

    if not assignment_expr_map:
        return axis_block_symbol_map

    changed = True
    while changed:
        changed = False
        for symbol, exprs in assignment_expr_map.items():
            if symbol in axis_block_symbol_map:
                continue
            candidate_blocks = set()
            for expr in exprs:
                for dep_name in extract_name_ids_transitive(expr, assignment_expr_map):
                    if dep_name == symbol:
                        continue
                    block_symbol = axis_block_symbol_map.get(dep_name)
                    if block_symbol is None:
                        continue
                    candidate_blocks.add(block_symbol)
            if len(candidate_blocks) != 1:
                continue
            axis_block_symbol_map[symbol] = next(iter(candidate_blocks))
            changed = True
    return axis_block_symbol_map


def extract_axis_total_symbol_map(
    func_node: ast.AST,
    *,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    axis_candidates: Set[str],
    param_names: Set[str],
) -> Dict[str, str]:
    axis_block_symbol_map = extract_axis_block_symbol_map(
        func_node,
        axis_candidates,
        assignment_expr_map=assignment_expr_map,
    )
    axis_total_symbol_map: Dict[str, str] = {}

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target_name = node.targets[0].id
        if target_name not in axis_candidates:
            continue
        if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Mod):
            continue
        candidate = pick_param_candidate(
            node.value.right,
            param_names=param_names,
            exclude={target_name},
            assignment_expr_map=assignment_expr_map,
        )
        if candidate is not None:
            axis_total_symbol_map[target_name] = candidate
        else:
            axis_total_symbol_map[target_name] = ast_to_text(node.value.right)

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        left_expr = node.left
        right_expr = node.comparators[0]

        left_axes = extract_axis_symbols_from_expr(left_expr, axis_candidates=axis_candidates)
        right_axes = extract_axis_symbols_from_expr(right_expr, axis_candidates=axis_candidates)

        left_only = left_axes - right_axes
        if left_only:
            candidate = pick_param_candidate(
                right_expr,
                param_names=param_names,
                exclude=set(left_only),
                assignment_expr_map=assignment_expr_map,
            )
            if candidate is not None:
                for axis_name in left_only:
                    axis_total_symbol_map.setdefault(axis_name, candidate)

        right_only = right_axes - left_axes
        if right_only:
            candidate = pick_param_candidate(
                left_expr,
                param_names=param_names,
                exclude=set(right_only),
                assignment_expr_map=assignment_expr_map,
            )
            if candidate is not None:
                for axis_name in right_only:
                    axis_total_symbol_map.setdefault(axis_name, candidate)

    for node in ast.walk(func_node):
        if not isinstance(node, ast.For):
            continue
        stop_expr, step_expr = extract_range_stop_and_step(node)
        if stop_expr is None:
            continue
        total_candidate = pick_param_candidate(
            stop_expr,
            param_names=param_names,
            exclude=set(),
            assignment_expr_map=assignment_expr_map,
        )
        if total_candidate is None:
            continue
        stop_names = extract_name_ids_transitive(stop_expr, assignment_expr_map)
        step_names = (extract_name_ids_transitive(step_expr, assignment_expr_map) if step_expr is not None else set())
        if not stop_names:
            if not step_names:
                continue
        for axis_name, block_symbol in axis_block_symbol_map.items():
            if axis_name in axis_total_symbol_map:
                continue
            if block_symbol not in stop_names and block_symbol not in step_names:
                continue
            axis_total_symbol_map[axis_name] = total_candidate

    changed = True
    while changed:
        changed = False
        for symbol, exprs in assignment_expr_map.items():
            if symbol in axis_total_symbol_map:
                continue
            symbol_block = axis_block_symbol_map.get(symbol)
            if symbol_block is None:
                continue
            candidate_totals = set()
            for expr in exprs:
                for dep_name in extract_name_ids_transitive(expr, assignment_expr_map):
                    if dep_name == symbol:
                        continue
                    total_symbol = axis_total_symbol_map.get(dep_name)
                    if total_symbol is None:
                        continue
                    dep_block = axis_block_symbol_map.get(dep_name)
                    if dep_block is not None and dep_block != symbol_block:
                        continue
                    candidate_totals.add(total_symbol)
            if len(candidate_totals) != 1:
                continue
            axis_total_symbol_map[symbol] = next(iter(candidate_totals))
            changed = True
    return axis_total_symbol_map
