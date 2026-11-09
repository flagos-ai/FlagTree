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
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .axis_length_resolver import classify_length_symbol
from .axis_semantic_schema import (AxisExtent, AxisSemanticInfo, AxisSemanticResult, AxisSplit, AxisTiling)
from .dynamic_source_utils import resolve_dynamic_source
from .load_semantics import (
    build_assignment_expr_map,
    build_symbol_user_expr_map,
    collect_axis_candidates,
    extract_axis_block_symbol_map,
    extract_axis_total_symbol_map,
    extract_name_depths_transitive,
    extract_name_ids,
    extract_name_ids_transitive,
    pick_param_candidate,
)
from .load_dependency_analyzer import collect_load_derived_symbols
from .schema import (AXIS_LENGTH_STATE_FIXED_COMPILE_TIME, AXIS_LENGTH_STATE_TUNABLE)
from .signature_analyzer import extract_signature_info

_VALID_AXIS_NAMES = ("x", "y", "z", "w", "v", "t")
_REDUCTION_FUNCS = ("sum", "xor_sum", "max", "min", "argmax", "argmin")


@dataclass(frozen=True)
class VvAxisInfoV2:
    axis_index: int
    length_expr: Optional[str]
    axis_symbol: Optional[str] = None
    state: str = "unknown"
    tunable_param: Optional[str] = None
    const_value: Optional[int] = None
    split_param: Optional[str] = None
    tiling_param: Optional[str] = None
    fixed_tiling_expr: Optional[str] = None
    is_low_dim: bool = False
    is_reduction: bool = False
    dynamic_source: str = "none"


@dataclass(frozen=True)
class VvAxisParseResultV2:
    axis_count: int
    axes: List[VvAxisInfoV2]
    source: str
    diagnostics: List[str]
    axis_length_exprs: Dict[str, str] = field(default_factory=dict)
    fixed_tiling_exprs: Dict[str, str] = field(default_factory=dict)
    axis_pid_dims: Dict[str, int] = field(default_factory=dict)
    inferred_keys: Dict[str, str] = field(default_factory=dict)
    split_params: Dict[str, str] = field(default_factory=dict)
    tiling_params: Dict[str, str] = field(default_factory=dict)
    axis_dynamic_sources: Dict[str, str] = field(default_factory=dict)
    low_dim_axes: List[str] = field(default_factory=list)
    reduction_axes: List[str] = field(default_factory=list)
    status: str = "ok"


@dataclass
class _SemanticAxisEvidence:
    axis_index: int
    axis_symbol: Optional[str] = None
    extent_candidates: List[Tuple[str, str, float]] = field(default_factory=list)
    split_candidates: List[Tuple[str, int, str, float]] = field(default_factory=list)
    tiling_candidates: List[Tuple[str, Optional[str], str, float]] = field(default_factory=list)
    fixed_tiling_candidates: List[Tuple[str, str, float]] = field(default_factory=list)
    is_low_dim: bool = False
    diagnostics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class _MakeBlockPtrSpec:
    shape_dims: List[str]
    block_shape_exprs: List[ast.AST]
    offset_exprs: List[ast.AST]


def _ast_to_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ast.unparse(node)


def _normalize_tunable_param_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    if not isinstance(symbol, str):
        return None
    if not symbol.isidentifier():
        return None
    return symbol


def _is_tl_call(node: ast.AST, name: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == name)


def _find_function_node(tree: ast.AST) -> ast.AST:
    if isinstance(tree, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return tree
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("Unable to locate function definition in VV parser v2 input AST.")


def _find_function_by_name(tree: ast.AST, function_name: Optional[str]) -> Optional[ast.AST]:
    if not function_name:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name == function_name:
            return node
    return None


def _resolve_function_node(
    func_ast: ast.AST,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> ast.AST:
    if isinstance(module_ast, ast.AST) and entry_function_name:
        resolved = _find_function_by_name(module_ast, entry_function_name)
        if resolved is not None:
            return resolved

    if isinstance(func_ast, ast.Module) and entry_function_name:
        resolved = _find_function_by_name(func_ast, entry_function_name)
        if resolved is not None:
            return resolved

    return _find_function_node(func_ast)


def _iter_walk_in_order(node: ast.AST):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _iter_walk_in_order(child)


def _get_arange_stop_expr(call_node: ast.Call) -> Optional[ast.AST]:
    if len(call_node.args) >= 2:
        return call_node.args[1]
    if len(call_node.args) == 1:
        return call_node.args[0]
    for keyword in call_node.keywords:
        if keyword.arg in ("end", "stop"):
            return keyword.value
    return None


def _collect_arange_stop_texts(expr: ast.AST) -> List[str]:
    values: List[str] = []
    seen = set()
    for node in _iter_walk_in_order(expr):
        if not _is_tl_call(node, "arange"):
            continue
        stop_expr = _get_arange_stop_expr(node)
        if stop_expr is None:
            continue
        text = _ast_to_text(stop_expr)
        if text in seen:
            continue
        seen.add(text)
        values.append(text)
    return values


def _extract_call_arg(
    call_node: ast.Call,
    keyword_name: str,
    positional_index: int,
) -> Optional[ast.AST]:
    for keyword in call_node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value
    if len(call_node.args) > positional_index:
        return call_node.args[positional_index]
    return None


def _extract_tuple_items(node: Optional[ast.AST]) -> List[ast.AST]:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return []
    return list(node.elts)


def _extract_make_block_ptr_spec(call_node: ast.Call) -> Optional[_MakeBlockPtrSpec]:
    shape_items = _extract_tuple_items(_extract_call_arg(call_node, "shape", 1))
    if not shape_items:
        return None
    block_shape_items = _extract_tuple_items(_extract_call_arg(call_node, "block_shape", 4))
    offset_items = _extract_tuple_items(_extract_call_arg(call_node, "offsets", 3))
    return _MakeBlockPtrSpec(
        shape_dims=[_ast_to_text(item) for item in shape_items],
        block_shape_exprs=block_shape_items,
        offset_exprs=offset_items,
    )


def _extract_make_block_ptr_shape(call_node: ast.Call) -> List[str]:
    spec = _extract_make_block_ptr_spec(call_node)
    if spec is None:
        return []
    return list(spec.shape_dims)


def _collect_make_block_ptr_specs(func_node: ast.AST) -> Dict[str, _MakeBlockPtrSpec]:
    result: Dict[str, _MakeBlockPtrSpec] = {}
    for node in _iter_walk_in_order(func_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not _is_tl_call(node.value, "make_block_ptr"):
            continue
        spec = _extract_make_block_ptr_spec(node.value)
        if spec is None:
            continue
        result[target.id] = spec
    return result


def _extract_target_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: List[str] = []
        for item in target.elts:
            names.extend(_extract_target_names(item))
        return names
    return []


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
            for child in expr.values:
                value = _resolve_static_bool_expr(child, known_values)
                if value is False:
                    return False
                if value is None:
                    maybe_unknown = True
            return None if maybe_unknown else True
        if isinstance(expr.op, ast.Or):
            maybe_unknown = False
            for child in expr.values:
                value = _resolve_static_bool_expr(child, known_values)
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


def _clone_assignment_expr_map(src: Mapping[str, Sequence[ast.AST]]) -> Dict[str, List[ast.AST]]:
    return {name: list(exprs) for name, exprs in src.items()}


def _append_unique_expr(values: List[ast.AST], expr: ast.AST) -> None:
    expr_dump = ast.dump(expr)
    for current in values:
        if ast.dump(current) == expr_dump:
            return
    values.append(expr)


def _merge_assignment_expr_map(
    dst: Dict[str, List[ast.AST]],
    src: Mapping[str, Sequence[ast.AST]],
) -> None:
    for name, exprs in src.items():
        bucket = dst.setdefault(name, [])
        for expr in exprs:
            _append_unique_expr(bucket, expr)


def _set_assignment_expr(
    assignment_expr_map: Dict[str, List[ast.AST]],
    target_names: Sequence[str],
    value_expr: ast.AST,
) -> None:
    for name in target_names:
        assignment_expr_map[name] = [value_expr]


def _visit_stmt_for_assignment_map(
    stmt: ast.stmt,
    assignment_expr_map: Dict[str, List[ast.AST]],
    known_values: Mapping[str, object],
) -> None:
    if isinstance(stmt, ast.Assign):
        target_names: List[str] = []
        for target in stmt.targets:
            target_names.extend(_extract_target_names(target))
        if target_names:
            _set_assignment_expr(assignment_expr_map, target_names, stmt.value)
        return

    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is None:
            return
        target_names = _extract_target_names(stmt.target)
        if target_names:
            _set_assignment_expr(assignment_expr_map, target_names, stmt.value)
        return

    if isinstance(stmt, ast.If):
        cond = _resolve_static_bool_expr(stmt.test, known_values)
        if cond is True:
            for child in stmt.body:
                _visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
            return
        if cond is False:
            for child in stmt.orelse:
                _visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
            return
        body_map = _clone_assignment_expr_map(assignment_expr_map)
        else_map = _clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            _visit_stmt_for_assignment_map(child, body_map, known_values)
        for child in stmt.orelse:
            _visit_stmt_for_assignment_map(child, else_map, known_values)
        assignment_expr_map.clear()
        _merge_assignment_expr_map(assignment_expr_map, body_map)
        _merge_assignment_expr_map(assignment_expr_map, else_map)
        return

    if isinstance(stmt, ast.For):
        loop_map = _clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            _visit_stmt_for_assignment_map(child, loop_map, known_values)
        for child in stmt.orelse:
            _visit_stmt_for_assignment_map(child, loop_map, known_values)
        _merge_assignment_expr_map(assignment_expr_map, loop_map)
        return

    if isinstance(stmt, ast.While):
        loop_map = _clone_assignment_expr_map(assignment_expr_map)
        for child in stmt.body:
            _visit_stmt_for_assignment_map(child, loop_map, known_values)
        for child in stmt.orelse:
            _visit_stmt_for_assignment_map(child, loop_map, known_values)
        _merge_assignment_expr_map(assignment_expr_map, loop_map)
        return

    if isinstance(stmt, ast.With):
        for child in stmt.body:
            _visit_stmt_for_assignment_map(child, assignment_expr_map, known_values)
        return


def _build_axis_symbol_to_length_map(func_node: ast.AST) -> Dict[str, str]:
    symbol_to_length: Dict[str, str] = {}
    assignment_expr: Dict[str, ast.AST] = {}
    for node in _iter_walk_in_order(func_node):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assignment_expr[target.id] = node.value
        if _is_tl_call(node.value, "arange"):
            stop_expr = _get_arange_stop_expr(node.value)
            if stop_expr is None:
                continue
            symbol_to_length[target.id] = _ast_to_text(stop_expr)
            continue
        arange_lengths = _collect_arange_stop_texts(node.value)
        if len(arange_lengths) == 1:
            symbol_to_length[target.id] = arange_lengths[0]

    changed = True
    while changed:
        changed = False
        for name, expr in assignment_expr.items():
            if name in symbol_to_length:
                continue
            if isinstance(expr, ast.Name) and expr.id in symbol_to_length:
                symbol_to_length[name] = symbol_to_length[expr.id]
                changed = True
    return symbol_to_length


def _iter_expr_closure(expr: ast.AST, assignment_expr_map: Mapping[str, Sequence[ast.AST]]):
    queue = [expr]
    seen = set()
    while queue:
        current = queue.pop(0)
        key = id(current)
        if key in seen:
            continue
        seen.add(key)
        yield current
        for node in _iter_walk_in_order(current):
            if not isinstance(node, ast.Name):
                continue
            assigned_exprs = assignment_expr_map.get(node.id, ())
            for assigned in assigned_exprs:
                queue.append(assigned)


def _is_slice_all(node: ast.AST) -> bool:
    return (isinstance(node, ast.Slice) and node.lower is None and node.upper is None and node.step is None)


def _is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _extract_axis_symbol_from_subscript_base(base_node: ast.AST) -> Optional[str]:
    if isinstance(base_node, ast.Name):
        return base_node.id
    if _is_tl_call(base_node, "arange") and isinstance(base_node, ast.Call):
        stop_expr = _get_arange_stop_expr(base_node)
        if stop_expr is not None:
            return _ast_to_text(stop_expr)
    arange_lengths = _collect_arange_stop_texts(base_node)
    if len(arange_lengths) == 1:
        return arange_lengths[0]
    return None


def _extract_subscript_axis_info(node: ast.Subscript) -> Optional[Tuple[int, str, int]]:
    symbol = _extract_axis_symbol_from_subscript_base(node.value)
    if symbol is None:
        return None
    if not isinstance(node.slice, ast.Tuple):
        return None
    dims = node.slice.elts
    if len(dims) < 2 or len(dims) > 4:
        return None
    axis_index = None
    for idx, dim in enumerate(dims):
        if _is_slice_all(dim):
            if axis_index is not None:
                return None
            axis_index = idx
            continue
        if _is_none_const(dim):
            continue
        return None
    if axis_index is None:
        return None
    return axis_index, symbol, len(dims)


def _extract_memory_pointer_expr(memory_node: ast.Call) -> Optional[ast.AST]:
    if memory_node.args:
        return memory_node.args[0]
    for keyword in memory_node.keywords:
        if keyword.arg in ("pointer", "ptr", "input"):
            return keyword.value
    return None


def _get_program_id_dim(node: ast.AST) -> Optional[int]:
    if not _is_tl_call(node, "program_id"):
        return None
    axis_dim = 0
    if len(node.args) > 0:
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            axis_dim = node.args[0].value
        else:
            return None
    for kw in node.keywords:
        if kw.arg != "axis":
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
            axis_dim = kw.value.value
        else:
            return None
        break
    return axis_dim


def _build_program_id_var_dims(func_node: ast.AST) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for node in _iter_walk_in_order(func_node):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        pid_dim = _get_program_id_dim(node.value)
        if pid_dim is None:
            continue
        result[node.targets[0].id] = pid_dim
    return result


def _extract_pid_dim_from_expr(expr: ast.AST, program_id_var_dims: Dict[str, int]) -> Optional[int]:
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in program_id_var_dims:
            return program_id_var_dims[node.id]
        pid_dim = _get_program_id_dim(node)
        if pid_dim is not None:
            return pid_dim
    return None


def _expr_has_axis_primitive(expr: ast.AST, program_id_var_dims: Dict[str, int]) -> bool:
    for node in ast.walk(expr):
        if _is_tl_call(node, "arange"):
            return True
        if isinstance(node, ast.Name) and node.id in program_id_var_dims:
            return True
        if _get_program_id_dim(node) is not None:
            return True
    return False


def _collect_offset_symbols_from_expr(
    ptr_expr: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    program_id_var_dims: Dict[str, int],
) -> List[str]:
    symbols: List[str] = []
    seen = set()
    queue: List[Tuple[str, int]] = []

    for node in _iter_walk_in_order(ptr_expr):
        if not isinstance(node, ast.Name):
            continue
        if node.id in seen:
            continue
        seen.add(node.id)
        queue.append((node.id, 0))

    while queue:
        name, depth = queue.pop(0)
        assigned_exprs = list(assignment_expr_map.get(name, ()))
        if not assigned_exprs:
            continue
        if any(_expr_has_axis_primitive(assigned, program_id_var_dims) for assigned in assigned_exprs):
            symbols.append(name)
            continue
        if depth >= 1:
            continue
        for assigned in assigned_exprs:
            for child in _iter_walk_in_order(assigned):
                if not isinstance(child, ast.Name):
                    continue
                if child.id in seen:
                    continue
                seen.add(child.id)
                queue.append((child.id, depth + 1))
    return symbols


def _collect_ptr_axis_symbols(
    ptr_expr: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    program_id_var_dims: Dict[str, int],
    axis_candidates: Optional[Set[str]] = None,
    value_axis_provenance: Optional[Dict[str, Dict[int, str]]] = None,
    provenance_visiting: Optional[Set[Tuple[str, int]]] = None,
) -> Tuple[List[int], Dict[int, str], Dict[int, bool]]:
    symbol_by_index: Dict[int, str] = {}
    is_low_dim_by_index: Dict[int, bool] = {}
    expr_owner_by_id = {}
    for name, exprs in assignment_expr_map.items():
        for expr in exprs:
            expr_owner_by_id[id(expr)] = name
    root_ptr_name = ptr_expr.id if isinstance(ptr_expr, ast.Name) else None
    axis_candidates = set(axis_candidates or ())
    value_axis_provenance = value_axis_provenance if value_axis_provenance is not None else {}
    provenance_visiting = provenance_visiting if provenance_visiting is not None else set()

    def _resolve_value_axis_symbol(symbol: str, axis_index: int) -> str:
        if symbol in axis_candidates:
            return symbol

        cached = value_axis_provenance.get(symbol, None)
        if cached is not None and axis_index in cached:
            return cached[axis_index]

        visiting_key = (symbol, axis_index)
        if visiting_key in provenance_visiting:
            return symbol

        provenance_visiting.add(visiting_key)
        try:
            for expr in assignment_expr_map.get(symbol, ()):
                base_expr = expr
                if isinstance(expr, ast.Call):
                    memory_expr = _extract_memory_pointer_expr(expr)
                    if memory_expr is not None:
                        base_expr = memory_expr
                axis_indices, resolved_symbols, _ = _collect_ptr_axis_symbols(
                    base_expr,
                    assignment_expr_map,
                    program_id_var_dims,
                    axis_candidates=axis_candidates,
                    value_axis_provenance=value_axis_provenance,
                    provenance_visiting=provenance_visiting,
                )
                if axis_index not in axis_indices:
                    continue
                resolved = resolved_symbols.get(axis_index, symbol)
                value_axis_provenance.setdefault(symbol, {})[axis_index] = resolved
                return resolved
        finally:
            provenance_visiting.discard(visiting_key)
        return symbol

    for current_expr in _iter_expr_closure(ptr_expr, assignment_expr_map):
        owner_symbol = expr_owner_by_id.get(id(current_expr))
        for node in _iter_walk_in_order(current_expr):
            if not isinstance(node, ast.Subscript):
                continue
            axis_info = _extract_subscript_axis_info(node)
            if axis_info is None:
                continue
            axis_index, symbol, ndim = axis_info
            if axis_index in symbol_by_index:
                continue
            if isinstance(symbol, str):
                symbol = _resolve_value_axis_symbol(symbol, axis_index)
            # For patterns like:
            #   row_offsets = ... + tl.arange(0, XBLOCK_SUB)[:, None]
            # prefer the assignment owner (`row_offsets`) over the inline
            # arange stop symbol (`XBLOCK_SUB`), so downstream split/tiling
            # extraction can trace program_id/loop evidence from the offset var.
            if (_is_tl_call(node.value, "arange") and isinstance(owner_symbol, str) and owner_symbol != root_ptr_name):
                symbol = owner_symbol
            elif (isinstance(owner_symbol, str) and owner_symbol != root_ptr_name and axis_candidates
                  and owner_symbol in axis_candidates and symbol not in axis_candidates):
                symbol = owner_symbol
            symbol_by_index[axis_index] = symbol
            is_low_dim_by_index[axis_index] = axis_index == (ndim - 1)

    if symbol_by_index:
        sorted_indices = sorted(symbol_by_index.keys())
        return sorted_indices, symbol_by_index, is_low_dim_by_index

    fallback_symbols = _collect_offset_symbols_from_expr(
        ptr_expr,
        assignment_expr_map,
        program_id_var_dims,
    )
    if not fallback_symbols:
        return [], {}, {}

    # Prefer high-confidence axis candidates collected from arange/mod/subscript
    # patterns when fallback symbols mix real vector offsets with base-pointer
    # symbols (e.g. `base + pid * stride + offset`).
    if axis_candidates:
        preferred_symbols = [symbol for symbol in fallback_symbols if symbol in axis_candidates]
        if preferred_symbols:
            fallback_symbols = preferred_symbols

    symbol_by_index = {idx: symbol for idx, symbol in enumerate(fallback_symbols)}
    is_low_dim_by_index = {idx: idx == (len(fallback_symbols) - 1) for idx in range(len(fallback_symbols))}
    return list(range(len(fallback_symbols))), symbol_by_index, is_low_dim_by_index


def _normalize_mask_extent_candidate_expr(
    rhs_expr: ast.AST,
    *,
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Optional[str]:
    if not (isinstance(rhs_expr, ast.Call) and isinstance(rhs_expr.func, ast.Name) and rhs_expr.func.id == "min"):
        return None

    args = list(rhs_expr.args)
    if len(args) < 2:
        return None

    dependency_names = _collect_symbol_dependency_names(symbol, assignment_expr_map)
    preferred_args: List[ast.AST] = []
    for arg in args:
        arg_names = extract_name_ids_transitive(arg, assignment_expr_map)
        if arg_names.intersection(dependency_names):
            continue
        preferred_args.append(arg)

    if len(preferred_args) != 1:
        return None
    return _ast_to_text(preferred_args[0])


def _collect_mask_extent_candidates_for_symbol(
    symbol: str,
    func_node: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    load_derived_symbols: Optional[Set[str]] = None,
) -> List[Tuple[str, str, float]]:
    candidates: List[Tuple[str, str, float]] = []
    seen = set()
    load_derived_symbols = load_derived_symbols or set()

    def _contains_symbol_direct(expr: ast.AST, target: str) -> bool:
        for child in ast.walk(expr):
            if isinstance(child, ast.Name) and child.id == target:
                return True
        return False

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
            continue
        if len(node.comparators) != 1:
            continue

        hit = False
        for current_expr in _iter_expr_closure(node.left, assignment_expr_map):
            for child in ast.walk(current_expr):
                if isinstance(child, ast.Name) and child.id == symbol:
                    hit = True
                    break
            if hit:
                break
        if not hit:
            continue

        # Reject value-domain comparisons (e.g. logits < 0) whose left side is
        # already load-derived. Such predicates should not be treated as axis
        # extent masks.
        if _expr_uses_symbols(node.left, load_derived_symbols):
            continue

        # Prefer direct offset-mask patterns (symbol appears directly on left).
        # Indirect dataflow hits are still kept but with lower confidence.
        is_direct_hit = _contains_symbol_direct(node.left, symbol)

        rhs_expr = _normalize_mask_extent_candidate_expr(
            node.comparators[0],
            symbol=symbol,
            assignment_expr_map=assignment_expr_map,
        ) or _ast_to_text(node.comparators[0])
        if rhs_expr in seen:
            continue
        seen.add(rhs_expr)
        source = "mask_direct" if is_direct_hit else "mask_indirect"
        confidence = 1.0 if is_direct_hit else 0.6
        candidates.append((rhs_expr, source, confidence))
    return candidates


def _extract_range_stop_and_step(node: ast.For) -> Tuple[Optional[ast.AST], Optional[ast.AST]]:
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


def _collect_loop_stop_candidates_for_symbol(
    symbol: str,
    func_node: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> List[Tuple[str, str, float]]:
    candidates: List[Tuple[str, str, float]] = []
    seen = set()
    dependency_names = _collect_symbol_dependency_names(symbol, assignment_expr_map)

    for node in ast.walk(func_node):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in dependency_names:
            continue
        stop_expr, _ = _extract_range_stop_and_step(node)
        if stop_expr is None:
            continue

        stop_text = _ast_to_text(stop_expr)
        if stop_text in seen:
            continue
        seen.add(stop_text)
        candidates.append((stop_text, "loop", 0.7))
    return candidates


def _collect_loop_steps(func_node: ast.AST) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for node in ast.walk(func_node):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        _, step_expr = _extract_range_stop_and_step(node)
        if step_expr is None:
            continue
        result[node.target.id] = _ast_to_text(step_expr)
    return result


def _extract_param_names(func_node: ast.AST) -> Set[str]:
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    args_node = func_node.args
    ordered_args = list(args_node.posonlyargs) + list(args_node.args) + list(args_node.kwonlyargs)
    return {arg.arg for arg in ordered_args if isinstance(arg, ast.arg)}


def _expr_uses_symbols(expr: ast.AST, symbols: Set[str]) -> bool:
    if not symbols:
        return False
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in symbols:
            return True
    return False


def _collect_symbol_dependency_names(
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
) -> Set[str]:
    names = {symbol}
    for expr in assignment_expr_map.get(symbol, ()):
        names.update(extract_name_ids_transitive(expr, assignment_expr_map))
    return names


def _extract_axis_symbols_from_expr(expr: ast.AST, axis_candidates: Set[str]) -> Set[str]:
    symbols: Set[str] = set()
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in axis_candidates:
            symbols.add(node.id)
            continue
        if not isinstance(node, ast.Subscript):
            continue
        axis_info = _extract_subscript_axis_info(node)
        if axis_info is None:
            continue
        _, symbol, _ = axis_info
        if symbol in axis_candidates:
            symbols.add(symbol)
    return symbols


def _extract_split_candidates_for_symbol(
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    program_id_var_dims: Dict[str, int],
    tunable_params: Set[str],
    symbol_user_expr_map: Optional[Mapping[str, Sequence[ast.AST]]] = None,
) -> List[Tuple[str, int, str, float]]:
    exprs = list(assignment_expr_map.get(symbol, ()))
    if symbol_user_expr_map is not None:
        exprs.extend(symbol_user_expr_map.get(symbol, ()))
    if not exprs:
        return []

    result: List[Tuple[str, int, str, float]] = []
    seen = set()
    for expr in exprs:
        for current_expr in _iter_expr_closure(expr, assignment_expr_map):
            for node in ast.walk(current_expr):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
                    continue
                sides = ((node.left, node.right), (node.right, node.left))
                for param_node, pid_side in sides:
                    if not isinstance(param_node, ast.Name):
                        continue
                    param_name = param_node.id
                    if param_name not in tunable_params:
                        continue
                    pid_dim = _extract_pid_dim_from_expr(pid_side, program_id_var_dims)
                    if pid_dim is None:
                        continue
                    key = (param_name, pid_dim)
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append((param_name, pid_dim, "program_id", 0.95))
    return result


def _extract_split_candidates_from_expr(
    expr: Optional[ast.AST],
    program_id_var_dims: Dict[str, int],
    tunable_params: Set[str],
) -> List[Tuple[str, int, str, float]]:
    if expr is None:
        return []
    result: List[Tuple[str, int, str, float]] = []
    seen = set()
    for node in ast.walk(expr):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        sides = ((node.left, node.right), (node.right, node.left))
        for param_node, pid_side in sides:
            if not isinstance(param_node, ast.Name):
                continue
            param_name = param_node.id
            if param_name not in tunable_params:
                continue
            pid_dim = _extract_pid_dim_from_expr(pid_side, program_id_var_dims)
            if pid_dim is None:
                continue
            key = (param_name, pid_dim)
            if key in seen:
                continue
            seen.add(key)
            result.append((param_name, pid_dim, "program_id", 0.95))
    return result


def _extract_make_block_ptr_tiling_param(
    expr: Optional[ast.AST],
    tunable_params: Set[str],
) -> Optional[str]:
    if expr is None:
        return None
    text = _normalize_tunable_param_symbol(_ast_to_text(expr))
    if text is None or text not in tunable_params:
        return None
    return text


def _extract_make_block_ptr_fixed_tiling_expr(
    expr: Optional[ast.AST],
    signature,
    provided_args: Mapping[str, object],
) -> Optional[str]:
    if expr is None:
        return None
    text = _ast_to_text(expr)
    state, _ = classify_length_symbol(
        text,
        signature=signature,
        provided_args=provided_args,
    )
    if state == AXIS_LENGTH_STATE_TUNABLE:
        return None
    if text and state == AXIS_LENGTH_STATE_FIXED_COMPILE_TIME:
        return text
    return None


def _extract_tiling_candidates_for_symbol(
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    tunable_params: Set[str],
    loop_steps: Dict[str, str],
) -> List[Tuple[str, Optional[str], str, float]]:
    exprs = list(assignment_expr_map.get(symbol, ()))
    if not exprs:
        return []

    result: List[Tuple[str, Optional[str], str, float]] = []
    seen = set()

    for expr in exprs:
        for current_expr in _iter_expr_closure(expr, assignment_expr_map):
            for stop_text in _collect_arange_stop_texts(current_expr):
                if stop_text not in tunable_params:
                    continue
                key = (stop_text, None, "arange")
                if key in seen:
                    continue
                seen.add(key)
                result.append((stop_text, None, "arange", 0.9))

            for node in ast.walk(current_expr):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
                    continue
                if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                    left_name = node.left.id
                    right_name = node.right.id
                    if left_name in loop_steps and right_name in tunable_params:
                        key = (right_name, left_name, "range")
                        if key not in seen:
                            seen.add(key)
                            result.append((right_name, left_name, "range", 0.75))
                    if right_name in loop_steps and left_name in tunable_params:
                        key = (left_name, right_name, "range")
                        if key not in seen:
                            seen.add(key)
                            result.append((left_name, right_name, "range", 0.75))
    return result


def _extract_fixed_tiling_candidates_for_symbol(
    symbol: str,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    signature,
    provided_args: Mapping[str, object],
) -> List[Tuple[str, str, float]]:
    exprs = list(assignment_expr_map.get(symbol, ()))
    if not exprs:
        return []

    result: List[Tuple[str, str, float]] = []
    seen = set()

    for expr in exprs:
        for current_expr in _iter_expr_closure(expr, assignment_expr_map):
            for stop_text in _collect_arange_stop_texts(current_expr):
                state, _ = classify_length_symbol(
                    stop_text,
                    signature=signature,
                    provided_args=provided_args,
                )
                if state != AXIS_LENGTH_STATE_FIXED_COMPILE_TIME:
                    continue
                key = (stop_text, "arange")
                if key in seen:
                    continue
                seen.add(key)
                result.append((stop_text, "arange", 0.9))
    return result


def _extract_range_start_expr(node: ast.For) -> Optional[ast.AST]:
    if not isinstance(node.iter, ast.Call):
        return None
    args = list(node.iter.args)
    if len(args) < 2:
        return None
    return args[0]


def _infer_axis_pid_dim_for_symbol(
    symbol: str,
    func_node: ast.AST,
    assignment_expr_map: Mapping[str, Sequence[ast.AST]],
    program_id_var_dims: Dict[str, int],
) -> Optional[int]:
    candidates: List[int] = []
    seen = set()

    for expr in assignment_expr_map.get(symbol, ()):
        pid_dim = _extract_pid_dim_from_expr(expr, program_id_var_dims)
        if pid_dim is None or pid_dim in seen:
            continue
        seen.add(pid_dim)
        candidates.append(pid_dim)

    dependency_names = _collect_symbol_dependency_names(symbol, assignment_expr_map)
    for node in ast.walk(func_node):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id not in dependency_names:
            continue
        start_expr = _extract_range_start_expr(node)
        pid_dim = _extract_pid_dim_from_expr(start_expr, program_id_var_dims)
        if pid_dim is None or pid_dim in seen:
            continue
        seen.add(pid_dim)
        candidates.append(pid_dim)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _select_best_split(candidates: Sequence[Tuple[str, int, str, float]], ) -> Tuple[AxisSplit, List[str]]:
    if not candidates:
        return AxisSplit(param=None, pid_dim=None, source="infer", confidence=0.0), []

    ordered = sorted(candidates, key=lambda item: (-item[3], item[0], item[1]))
    best = ordered[0]
    diagnostics: List[str] = []
    distinct = {(param, pid_dim) for param, pid_dim, _, _ in ordered}
    if len(distinct) > 1:
        diagnostics.append("multiple split candidates resolved, selected {}".format(best[0]))
    return AxisSplit(param=best[0], pid_dim=best[1], source=best[2], confidence=best[3]), diagnostics


def _select_best_tiling(
    candidates: Sequence[Tuple[str, Optional[str], str, float]],
    split_param: Optional[str],
) -> Tuple[AxisTiling, List[str]]:
    if not candidates:
        return AxisTiling(param=None, loop_var=None, source="infer", confidence=0.0), []

    ordered = sorted(candidates, key=lambda item: (-item[3], item[0]))
    diagnostics: List[str] = []
    filtered = [item for item in ordered if item[0] != split_param]
    if filtered:
        best = filtered[0]
    else:
        best = ordered[0]
        diagnostics.append("tiling candidate {} equals split candidate, use single-parameter mode".format(best[0]))
        return AxisTiling(param=None, loop_var=best[1], source=best[2], confidence=best[3]), diagnostics

    distinct = {param for param, _, _, _ in ordered}
    if len(distinct) > 1:
        diagnostics.append("multiple tiling candidates resolved, selected {}".format(best[0]))
    return AxisTiling(param=best[0], loop_var=best[1], source=best[2], confidence=best[3]), diagnostics


def _select_best_fixed_tiling_expr(candidates: Sequence[Tuple[str, str, float]], ) -> Tuple[Optional[str], List[str]]:
    if not candidates:
        return None, []

    ordered = sorted(candidates, key=lambda item: (-item[2], item[0]))
    best = ordered[0]
    diagnostics: List[str] = []
    distinct = {expr for expr, _, _ in ordered}
    if len(distinct) > 1:
        diagnostics.append("multiple fixed tiling candidates resolved, selected {}".format(best[0]))
    return best[0], diagnostics


def _extent_priority(source: str) -> int:
    if source == "mask_direct":
        return 1
    if source == "mask_indirect":
        return 2
    if source == "infer_total":
        return 3
    if source == "loop":
        return 4
    if source == "make_block_ptr":
        return 5
    return 5


def _select_best_extent(
    candidates: Sequence[Tuple[str, str, float]],
    split_param: Optional[str],
    tiling_param: Optional[str],
    signature,
    provided_args: Mapping[str, object],
    axis_symbol: Optional[str] = None,
    load_derived_symbols: Optional[Set[str]] = None,
) -> Tuple[AxisExtent, List[str]]:
    load_derived_symbols = load_derived_symbols or set()

    if not candidates:
        dynamic_source = resolve_dynamic_source(
            axis_symbol=axis_symbol,
            length_expr=None,
            load_derived_symbols=load_derived_symbols,
        )
        return (
            AxisExtent(
                expr=None,
                state="unknown",
                const_value=None,
                source="infer",
                confidence=0.0,
                dynamic_source=dynamic_source,
            ),
            ["no extent candidate found"],
        )

    dedup: Dict[str, Tuple[str, str, float]] = {}
    for expr, source, confidence in candidates:
        old = dedup.get(expr)
        if old is None or confidence > old[2]:
            dedup[expr] = (expr, source, confidence)
    ordered = sorted(
        dedup.values(),
        key=lambda item: (_extent_priority(item[1]), -item[2], item[0]),
    )

    tile_params = {item for item in (split_param, tiling_param) if item}
    filtered = [item for item in ordered if item[0] not in tile_params]

    diagnostics: List[str] = []
    if filtered:
        best = filtered[0]
    else:
        best = ordered[0]
        diagnostics.append("extent fallback to tile parameter {} because no better candidate".format(best[0]))

    state, const_value = classify_length_symbol(
        best[0],
        signature=signature,
        provided_args=provided_args,
    )
    dynamic_source = resolve_dynamic_source(
        axis_symbol=axis_symbol,
        length_expr=best[0],
        load_derived_symbols=load_derived_symbols,
    )
    return (
        AxisExtent(
            expr=best[0],
            state=state,
            const_value=const_value,
            source=best[1],
            confidence=best[2],
            dynamic_source=dynamic_source,
        ),
        diagnostics,
    )


def _resolve_tunable_params(signature, provided_args: Mapping[str, object],
                            hints: Optional[Mapping[str, object]]) -> Set[str]:
    constexpr_names = set(signature.constexpr_names())
    missing_constexpr = {name for name in constexpr_names if name not in dict(provided_args or {}).keys()}

    explicit_tunable = set()
    if hints and isinstance(hints.get("tunable_parameter"), list):
        for item in hints.get("tunable_parameter"):
            if isinstance(item, str):
                explicit_tunable.add(item)

    return missing_constexpr | explicit_tunable


def _collect_reduction_axis_indices(func_node: ast.AST, axis_count: int) -> List[int]:
    if axis_count <= 0:
        return []

    result: List[int] = []
    seen = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl" and node.func.attr in _REDUCTION_FUNCS):
            continue

        axis_node = None
        if len(node.args) >= 2:
            axis_node = node.args[1]
        else:
            for kw in node.keywords:
                if kw.arg == "axis":
                    axis_node = kw.value
                    break
        if axis_node is None:
            continue

        axis_idx = None
        if isinstance(axis_node, ast.Constant) and isinstance(axis_node.value, int):
            axis_idx = axis_node.value
        elif (isinstance(axis_node, ast.UnaryOp) and isinstance(axis_node.op, ast.USub)
              and isinstance(axis_node.operand, ast.Constant) and isinstance(axis_node.operand.value, int)):
            axis_idx = -axis_node.operand.value

        if axis_idx is None:
            continue
        if axis_idx < 0:
            axis_idx = axis_count + axis_idx
        if axis_idx < 0 or axis_idx >= axis_count:
            continue
        if axis_idx in seen:
            continue
        seen.add(axis_idx)
        result.append(axis_idx)
    return result


def _axis_sort_key(axis_name: str) -> Tuple[int, str]:
    try:
        axis_index = _VALID_AXIS_NAMES.index(axis_name)
    except ValueError:
        axis_index = len(_VALID_AXIS_NAMES)
    return (axis_index, axis_name)


def _choose_best_site_evidence(
        site_evidences: Sequence[Dict[int, _SemanticAxisEvidence]]) -> Dict[int, _SemanticAxisEvidence]:
    if not site_evidences:
        return {}

    extent_source_score = {
        "mask": 5,
        "infer_total": 4,
        "loop": 3,
        "make_block_ptr": 2,
        "load": 1,
    }

    def _best_extent_quality(evidence: _SemanticAxisEvidence) -> Tuple[int, float]:
        if not evidence.extent_candidates:
            return (0, 0.0)
        best_source_score = 0
        best_confidence = 0.0
        for _, source, confidence in evidence.extent_candidates:
            source_score = extent_source_score.get(source, 0)
            if (source_score, confidence) > (best_source_score, best_confidence):
                best_source_score = source_score
                best_confidence = confidence
        return (best_source_score, best_confidence)

    def _score(site: Dict[int, _SemanticAxisEvidence]) -> Tuple[int, int, float, int]:
        axis_count = len(site)
        resolved_extent_axes = sum(1 for evidence in site.values() if len(evidence.extent_candidates) > 0)
        extent_quality_score = sum(_best_extent_quality(evidence)[0] for evidence in site.values())
        extent_confidence_score = sum(_best_extent_quality(evidence)[1] for evidence in site.values())
        return (
            resolved_extent_axes,
            extent_quality_score,
            extent_confidence_score,
            axis_count,
        )

    best = site_evidences[0]
    best_score = _score(best)
    for site in site_evidences[1:]:
        score = _score(site)
        if score > best_score:
            best = site
            best_score = score
    return best


def parse_vv_axis_semantic_v2(
    func_ast: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
    hints: Optional[Mapping[str, object]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> AxisSemanticResult:
    """
    Unified axis semantic parser (v2): extent/split/tiling/low_dim/reduction.
    """
    func_node = _resolve_function_node(
        func_ast,
        module_ast=module_ast,
        entry_function_name=entry_function_name,
    )
    provided_args = dict(provided_args or {})
    load_derived_symbols: Set[str] = set()
    try:
        analysis_root = (module_ast if isinstance(module_ast, ast.AST) and entry_function_name else func_node)
        load_derived_symbols = collect_load_derived_symbols(
            analysis_root,
            provided_args=provided_args,
            entry_function_name=entry_function_name,
        )
    except Exception as exc:
        load_derived_symbols = set()
        diagnostics = ["load-dependency analysis failed: {}".format(type(exc).__name__)]
    else:
        diagnostics: List[str] = []

    signature = extract_signature_info(func_node)
    tunable_params = _resolve_tunable_params(signature, provided_args, hints)
    param_names = _extract_param_names(func_node)

    make_block_ptr_specs = _collect_make_block_ptr_specs(func_node)
    assignment_expr_map = build_assignment_expr_map(
        func_node,
        provided_args=provided_args,
    )
    symbol_user_expr_map = build_symbol_user_expr_map(assignment_expr_map)
    symbol_to_length = _build_axis_symbol_to_length_map(func_node)
    program_id_var_dims = _build_program_id_var_dims(func_node)
    loop_steps = _collect_loop_steps(func_node)
    axis_candidates = collect_axis_candidates(func_node, param_names=param_names)
    axis_total_symbol_map = extract_axis_total_symbol_map(
        func_node,
        assignment_expr_map=assignment_expr_map,
        axis_candidates=axis_candidates,
        param_names=param_names,
    )

    source_tags = set()
    site_evidences: List[Dict[int, _SemanticAxisEvidence]] = []

    def _ensure_evidence(
        site: Dict[int, _SemanticAxisEvidence],
        axis_index: int,
    ) -> _SemanticAxisEvidence:
        if axis_index not in site:
            site[axis_index] = _SemanticAxisEvidence(axis_index=axis_index)
        return site[axis_index]

    def _append_extent(
        site: Dict[int, _SemanticAxisEvidence],
        axis_index: int,
        expr: str,
        source: str,
        confidence: float,
    ) -> None:
        if not expr:
            return
        evidence = _ensure_evidence(site, axis_index)
        key = (expr, source)
        for item in evidence.extent_candidates:
            if (item[0], item[1]) == key:
                return
        evidence.extent_candidates.append((expr, source, confidence))

    def _append_split(
        site: Dict[int, _SemanticAxisEvidence],
        axis_index: int,
        param: str,
        pid_dim: int,
        source: str,
        confidence: float,
    ) -> None:
        evidence = _ensure_evidence(site, axis_index)
        key = (param, pid_dim, source)
        for item in evidence.split_candidates:
            if (item[0], item[1], item[2]) == key:
                return
        evidence.split_candidates.append((param, pid_dim, source, confidence))

    def _append_tiling(
        site: Dict[int, _SemanticAxisEvidence],
        axis_index: int,
        param: str,
        loop_var: Optional[str],
        source: str,
        confidence: float,
    ) -> None:
        evidence = _ensure_evidence(site, axis_index)
        key = (param, loop_var, source)
        for item in evidence.tiling_candidates:
            if (item[0], item[1], item[2]) == key:
                return
        evidence.tiling_candidates.append((param, loop_var, source, confidence))

    def _append_fixed_tiling(
        site: Dict[int, _SemanticAxisEvidence],
        axis_index: int,
        expr: str,
        source: str,
        confidence: float,
    ) -> None:
        if not expr:
            return
        evidence = _ensure_evidence(site, axis_index)
        key = (expr, source)
        for item in evidence.fixed_tiling_candidates:
            if (item[0], item[1]) == key:
                return
        evidence.fixed_tiling_candidates.append((expr, source, confidence))

    def _append_make_block_ptr_spec(
        site: Dict[int, _SemanticAxisEvidence],
        spec: _MakeBlockPtrSpec,
    ) -> None:
        if spec.shape_dims:
            source_tags.add("make_block_ptr")
        for axis_index, dim in enumerate(spec.shape_dims):
            _append_extent(site, axis_index, dim, "make_block_ptr", 0.6)
        for axis_index, offset_expr in enumerate(spec.offset_exprs):
            for param, pid_dim, source, confidence in _extract_split_candidates_from_expr(
                    offset_expr,
                    program_id_var_dims,
                    tunable_params,
            ):
                _append_split(site, axis_index, param, pid_dim, source, confidence)
        for axis_index, block_shape_expr in enumerate(spec.block_shape_exprs):
            param = _extract_make_block_ptr_tiling_param(
                block_shape_expr,
                tunable_params,
            )
            if param is not None:
                _append_tiling(site, axis_index, param, None, "make_block_ptr", 0.85)
                continue
            fixed_expr = _extract_make_block_ptr_fixed_tiling_expr(
                block_shape_expr,
                signature,
                provided_args,
            )
            if fixed_expr is not None:
                _append_fixed_tiling(
                    site,
                    axis_index,
                    fixed_expr,
                    "make_block_ptr",
                    0.8,
                )

    for node in _iter_walk_in_order(func_node):
        if _is_tl_call(node, "make_block_ptr"):
            current_site: Dict[int, _SemanticAxisEvidence] = {}
            spec = _extract_make_block_ptr_spec(node)
            if spec is not None:
                _append_make_block_ptr_spec(current_site, spec)
            if current_site:
                site_evidences.append(current_site)
            continue

        is_load = _is_tl_call(node, "load")
        is_store = _is_tl_call(node, "store")
        if not is_load and not is_store:
            continue

        current_site: Dict[int, _SemanticAxisEvidence] = {}
        ptr_expr = _extract_memory_pointer_expr(node)
        if ptr_expr is None:
            continue

        source_tags.add("load" if is_load else "store")

        if isinstance(ptr_expr, ast.Name) and ptr_expr.id in make_block_ptr_specs:
            _append_make_block_ptr_spec(current_site, make_block_ptr_specs[ptr_expr.id])
            if current_site:
                site_evidences.append(current_site)
            continue

        if _is_tl_call(ptr_expr, "make_block_ptr"):
            spec = _extract_make_block_ptr_spec(ptr_expr)
            if spec is not None:
                _append_make_block_ptr_spec(current_site, spec)
            if current_site:
                site_evidences.append(current_site)
            continue

        axis_indices, symbol_by_index, is_low_dim_by_index = _collect_ptr_axis_symbols(
            ptr_expr,
            assignment_expr_map,
            program_id_var_dims,
            axis_candidates=axis_candidates,
        )

        if not axis_indices:
            arange_lengths: List[str] = []
            seen = set()
            for current_expr in _iter_expr_closure(ptr_expr, assignment_expr_map):
                for stop_text in _collect_arange_stop_texts(current_expr):
                    if stop_text in seen:
                        continue
                    seen.add(stop_text)
                    arange_lengths.append(stop_text)
            for axis_index, length_expr in enumerate(arange_lengths):
                _append_extent(current_site, axis_index, length_expr, "load", 0.5)
            if current_site:
                site_evidences.append(current_site)
            continue

        for axis_index in axis_indices:
            evidence = _ensure_evidence(current_site, axis_index)
            symbol = symbol_by_index.get(axis_index)
            if symbol and evidence.axis_symbol is None:
                evidence.axis_symbol = symbol
            evidence.is_low_dim = evidence.is_low_dim or is_low_dim_by_index.get(axis_index, False)

            if not symbol:
                continue

            inferred_total_expr = axis_total_symbol_map.get(symbol, None)
            if inferred_total_expr:
                _append_extent(current_site, axis_index, inferred_total_expr, "infer_total", 0.85)

            for expr, source, confidence in _collect_mask_extent_candidates_for_symbol(
                    symbol,
                    func_node,
                    assignment_expr_map,
                    load_derived_symbols,
            ):
                _append_extent(current_site, axis_index, expr, source, confidence)

            for expr, source, confidence in _collect_loop_stop_candidates_for_symbol(
                    symbol,
                    func_node,
                    assignment_expr_map,
            ):
                _append_extent(current_site, axis_index, expr, source, confidence)

            fallback_expr = symbol_to_length.get(symbol)
            if fallback_expr:
                _append_extent(current_site, axis_index, fallback_expr, "load", 0.4)

            for param, pid_dim, source, confidence in _extract_split_candidates_for_symbol(
                    symbol,
                    assignment_expr_map,
                    program_id_var_dims,
                    tunable_params,
                    symbol_user_expr_map=symbol_user_expr_map,
            ):
                _append_split(current_site, axis_index, param, pid_dim, source, confidence)

            for param, loop_var, source, confidence in _extract_tiling_candidates_for_symbol(
                    symbol,
                    assignment_expr_map,
                    tunable_params,
                    loop_steps,
            ):
                _append_tiling(current_site, axis_index, param, loop_var, source, confidence)

            for expr, source, confidence in _extract_fixed_tiling_candidates_for_symbol(
                    symbol,
                    assignment_expr_map,
                    signature,
                    provided_args,
            ):
                _append_fixed_tiling(current_site, axis_index, expr, source, confidence)

        if current_site:
            site_evidences.append(current_site)

    evidence_by_index = _choose_best_site_evidence(site_evidences)
    if not evidence_by_index:
        return AxisSemanticResult(
            axes={},
            axis_length_exprs={},
            fixed_tiling_exprs={},
            axis_pid_dims={},
            inferred_keys={},
            split_params={},
            tiling_params={},
            low_dim_axes=[],
            reduction_axes=[],
            status="failed",
            diagnostics=["no axis information resolved from tl.load/tl.store/make_block_ptr"],
        )

    sorted_axis_indices = sorted(evidence_by_index.keys())
    axis_name_by_index: Dict[int, str] = {}
    for order, axis_index in enumerate(sorted_axis_indices):
        if order < len(_VALID_AXIS_NAMES):
            axis_name_by_index[axis_index] = _VALID_AXIS_NAMES[order]
        else:
            axis_name_by_index[axis_index] = "x{}".format(order)

    reduction_indices = _collect_reduction_axis_indices(func_node, len(sorted_axis_indices))
    reduction_base_axes = {
        axis_name_by_index[sorted_axis_indices[item]]
        for item in reduction_indices
        if 0 <= item < len(sorted_axis_indices)
    }

    axes: Dict[str, AxisSemanticInfo] = {}
    axis_length_exprs: Dict[str, str] = {}
    fixed_tiling_exprs: Dict[str, str] = {}
    axis_pid_dims: Dict[str, int] = {}
    inferred_keys: Dict[str, str] = {}
    split_params: Dict[str, str] = {}
    tiling_params: Dict[str, str] = {}
    low_dim_axes: List[str] = []
    reduction_axes: List[str] = []

    for axis_index in sorted_axis_indices:
        evidence = evidence_by_index[axis_index]
        base_axis_name = axis_name_by_index[axis_index]
        is_reduction = base_axis_name in reduction_base_axes
        axis_name = base_axis_name

        split, split_diag = _select_best_split(evidence.split_candidates)
        tiling, tiling_diag = _select_best_tiling(evidence.tiling_candidates, split.param)
        fixed_tiling_expr, fixed_tiling_diag = _select_best_fixed_tiling_expr(evidence.fixed_tiling_candidates)
        extent, extent_diag = _select_best_extent(
            evidence.extent_candidates,
            split.param,
            tiling.param,
            signature,
            provided_args,
            axis_symbol=evidence.axis_symbol,
            load_derived_symbols=load_derived_symbols,
        )

        axis_diags: List[str] = []
        axis_diags.extend(split_diag)
        axis_diags.extend(tiling_diag)
        axis_diags.extend(fixed_tiling_diag)
        axis_diags.extend(extent_diag)
        axis_diags.extend(evidence.diagnostics)

        if extent.expr is not None:
            axis_length_exprs[axis_name] = extent.expr
            inferred_keys[axis_name] = extent.expr
        pid_dim = split.pid_dim
        if pid_dim is None and evidence.axis_symbol is not None:
            pid_dim = _infer_axis_pid_dim_for_symbol(
                evidence.axis_symbol,
                func_node,
                assignment_expr_map,
                program_id_var_dims,
            )
        if pid_dim is not None:
            axis_pid_dims[axis_name] = pid_dim
        if split.param is not None:
            split_params[axis_name] = split.param
        if tiling.param is not None:
            tiling_params[axis_name] = tiling.param
        if fixed_tiling_expr is not None:
            fixed_tiling_exprs[axis_name] = fixed_tiling_expr

        if evidence.is_low_dim:
            low_dim_axes.append(axis_name)
        if is_reduction:
            reduction_axes.append(axis_name)

        axes[axis_name] = AxisSemanticInfo(
            axis_name=axis_name,
            extent=extent,
            split=split,
            tiling=AxisTiling(
                param=tiling.param,
                loop_var=tiling.loop_var,
                source=tiling.source,
                confidence=tiling.confidence,
                fixed_expr=fixed_tiling_expr,
            ),
            is_low_dim=evidence.is_low_dim,
            is_reduction=is_reduction,
            diagnostics=axis_diags,
        )

    if len(low_dim_axes) != len(set(low_dim_axes)):
        diagnostics.append("duplicate low_dim axis detected")
        low_dim_axes = list(dict.fromkeys(low_dim_axes))
    if len(reduction_axes) != len(set(reduction_axes)):
        diagnostics.append("duplicate reduction axis detected")
        reduction_axes = list(dict.fromkeys(reduction_axes))

    for axis_info in axes.values():
        if axis_info.diagnostics:
            diagnostics.extend(["{}: {}".format(axis_info.axis_name, item) for item in axis_info.diagnostics])

    status = "ok"
    if diagnostics or any(info.extent.expr is None for info in axes.values()):
        status = "partial"

    if not axes:
        status = "failed"

    return AxisSemanticResult(
        axes=axes,
        axis_length_exprs=axis_length_exprs,
        fixed_tiling_exprs=fixed_tiling_exprs,
        axis_pid_dims=axis_pid_dims,
        inferred_keys=inferred_keys,
        split_params=split_params,
        tiling_params=tiling_params,
        low_dim_axes=low_dim_axes,
        reduction_axes=reduction_axes,
        status=status,
        diagnostics=diagnostics,
    )


def parse_vv_axis_info_v2(
    func_ast: ast.AST,
    provided_args: Optional[Mapping[str, object]] = None,
    hints: Optional[Mapping[str, object]] = None,
    module_ast: Optional[ast.AST] = None,
    entry_function_name: Optional[str] = None,
) -> VvAxisParseResultV2:
    semantic_result = parse_vv_axis_semantic_v2(
        func_ast,
        provided_args=provided_args,
        hints=hints,
        module_ast=module_ast,
        entry_function_name=entry_function_name,
    )

    ordered_axis_names = sorted(semantic_result.axes.keys(), key=_axis_sort_key)
    legacy_axes: List[VvAxisInfoV2] = []
    axis_dynamic_sources: Dict[str, str] = {}

    for axis_index, axis_name in enumerate(ordered_axis_names):
        semantic_axis = semantic_result.axes[axis_name]
        extent = semantic_axis.extent
        tunable_param = None
        if extent.state == AXIS_LENGTH_STATE_TUNABLE:
            tunable_param = _normalize_tunable_param_symbol(extent.expr)
        axis_dynamic_sources[axis_name] = extent.dynamic_source

        legacy_axes.append(
            VvAxisInfoV2(
                axis_index=axis_index,
                length_expr=extent.expr,
                axis_symbol=None,
                state=extent.state,
                tunable_param=tunable_param,
                const_value=extent.const_value,
                split_param=semantic_axis.split.param,
                tiling_param=semantic_axis.tiling.param,
                fixed_tiling_expr=semantic_axis.tiling.fixed_expr,
                is_low_dim=semantic_axis.is_low_dim,
                is_reduction=semantic_axis.is_reduction,
                dynamic_source=extent.dynamic_source,
            ))

    if not semantic_result.axes:
        source = "none"
    else:
        extent_sources = {axis_info.extent.source for axis_info in semantic_result.axes.values()}
        if extent_sources == {"make_block_ptr"}:
            source = "make_block_ptr"
        elif extent_sources.intersection({"mask", "infer_total", "loop", "load"}):
            source = "load"
        else:
            source = "axis_semantic"

    return VvAxisParseResultV2(
        axis_count=len(legacy_axes),
        axes=legacy_axes,
        source=source,
        diagnostics=list(semantic_result.diagnostics),
        axis_length_exprs=dict(semantic_result.axis_length_exprs),
        fixed_tiling_exprs=dict(semantic_result.fixed_tiling_exprs),
        axis_pid_dims=dict(semantic_result.axis_pid_dims),
        inferred_keys=dict(semantic_result.inferred_keys),
        split_params=dict(semantic_result.split_params),
        tiling_params=dict(semantic_result.tiling_params),
        axis_dynamic_sources=axis_dynamic_sources,
        low_dim_axes=list(semantic_result.low_dim_axes),
        reduction_axes=list(semantic_result.reduction_axes),
        status=semantic_result.status,
    )
