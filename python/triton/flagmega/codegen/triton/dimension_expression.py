# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Emit FlagMega :class:`Dimension`/:class:`LogicalExpr` trees as Triton."""

from __future__ import annotations

from collections.abc import Mapping

from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.dim_expr import DimConst, DimExpr, DimVar, Dimension, UnknownDim
from triton.flagmega.ir.logical_expr import (
    BoolConst,
    DimCompare,
    LogicalExpr,
    LogicalNot,
    LogicalOp,
)

_COMPARISON_SOURCE = {
    "equal": "==",
    "not_equal": "!=",
    "less_than": "<",
    "less_or_equal": "<=",
    "greater_than": ">",
    "greater_or_equal": ">=",
}


def emit_dimension(value: Dimension, *, symbols: Mapping[str, str] | None = None) -> str:
    """Render the complete supported DimExpr algebra without evaluating it."""

    if isinstance(value, DimConst):
        return str(value.fixed)
    if isinstance(value, DimVar):
        if symbols is not None:
            try:
                return symbols[value.symbol]
            except KeyError as error:
                raise CodegenError(f"Unbound runtime dimension symbol {value.symbol!r}.") from error
        return value.symbol
    if isinstance(value, UnknownDim):
        raise CodegenError("An unknown dimension cannot be emitted into Triton source.")
    if not isinstance(value, DimExpr):  # pragma: no cover - Dimension is closed.
        raise CodegenError(f"Unsupported dimension node {type(value).__name__}.")
    operands = tuple(emit_dimension(operand, symbols=symbols) for operand in value.operands)
    if value.op == "add":
        return "(" + " + ".join(operands) + ")"
    if value.op == "mul":
        return "(" + " * ".join(operands) + ")"
    if value.op == "floor_div":
        return f"(({operands[0]}) // ({operands[1]}))"
    if value.op == "ceil_div":
        return f"(-(-({operands[0]}) // ({operands[1]})))"
    if value.op == "mod":
        return f"(({operands[0]}) % ({operands[1]}))"
    if value.op == "pow":
        return f"(({operands[0]}) ** ({operands[1]}))"
    if value.op in {"min", "max"}:
        function = "tl.minimum" if value.op == "min" else "tl.maximum"
        result = operands[0]
        for operand in operands[1:]:
            result = f"{function}({result}, {operand})"
        return result
    if value.op == "abs":
        return f"tl.abs({operands[0]})"
    if value.op == "clamp":
        return (
            f"tl.minimum(tl.maximum({operands[0]}, {operands[1]}), "
            f"{operands[2]})"
        )
    if value.op == "positive":
        return (
            f"tl.where(({operands[0]}) >= 0, {operands[0]}, "
            f"({operands[0]}) + ({operands[1]}))"
        )
    if value.op.startswith("select_"):
        comparison = {
            "select_eq": "==",
            "select_ne": "!=",
            "select_lt": "<",
            "select_le": "<=",
            "select_gt": ">",
            "select_ge": ">=",
        }[value.op]
        return (
            f"tl.where(({operands[0]}) {comparison} ({operands[1]}), "
            f"{operands[2]}, {operands[3]})"
        )
    raise CodegenError(f"Unsupported dimension expression op {value.op!r}.")


def emit_logical(value: LogicalExpr, *, symbols: Mapping[str, str] | None = None) -> str:
    """Render a boolean expression over dimensions as Triton source.

    The result is a scalar predicate, so it composes with ordinary Triton
    boolean operators and needs no block shape.
    """

    if isinstance(value, BoolConst):
        return "True" if value.constant else "False"
    if isinstance(value, DimCompare):
        comparison = _COMPARISON_SOURCE[value.compare.value]
        lhs = emit_dimension(value.lhs, symbols=symbols)
        rhs = emit_dimension(value.rhs, symbols=symbols)
        return f"(({lhs}) {comparison} ({rhs}))"
    if isinstance(value, LogicalNot):
        return f"(not {emit_logical(value.operand, symbols=symbols)})"
    if isinstance(value, LogicalOp):
        joiner = " and " if value.op == "and" else " or "
        return "(" + joiner.join(
            emit_logical(operand, symbols=symbols) for operand in value.operands) + ")"
    raise CodegenError(f"Unsupported logical expression node {type(value).__name__}.")


__all__ = ["emit_dimension", "emit_logical"]
