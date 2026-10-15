# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable boolean expressions over integer dimensions.

``Dimension`` describes an integer-valued expression; this module adds its
boolean sibling. Both are plain immutable value trees that round-trip through
Python IR, mirroring nncase's ``Dimension``/``BaseExpr`` relation where a
comparison is an expression node rather than an operator in the graph.

``DimCompare`` compares two dimensions and yields a boolean, so a predicate
such as "this is the last loop iteration" stays an expression instead of
becoming stored data that has to be loaded back at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.dim_expr import (
    DimCompareOp,
    Dimension,
    DimensionLike,
    dim,
    equivalent_dim,
    evaluate_dim,
    simplify_dim,
    substitute_dim,
)

_COMPARISONS = {
    DimCompareOp.EQUAL: ("==", lambda lhs, rhs: lhs == rhs),
    DimCompareOp.NOT_EQUAL: ("!=", lambda lhs, rhs: lhs != rhs),
    DimCompareOp.LESS_THAN: ("<", lambda lhs, rhs: lhs < rhs),
    DimCompareOp.LESS_OR_EQUAL: ("<=", lambda lhs, rhs: lhs <= rhs),
    DimCompareOp.GREATER_THAN: (">", lambda lhs, rhs: lhs > rhs),
    DimCompareOp.GREATER_OR_EQUAL: (">=", lambda lhs, rhs: lhs >= rhs),
}


class LogicalExpr:
    """Base class for boolean-valued expressions over dimensions."""

    @property
    def is_fixed(self) -> bool:
        """Whether this predicate is decidable without runtime bindings."""
        return self.value is not None

    @property
    def value(self) -> bool | None:
        """The constant truth value, when it is provable."""
        return None

    def evaluate(self, bindings: Mapping[str, int]) -> bool:
        raise NotImplementedError

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> LogicalExpr:
        raise NotImplementedError

    def simplify(self) -> LogicalExpr:
        return self

    def to_data(self) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def from_data(data: Mapping[str, Any]) -> LogicalExpr:
        kind = data.get("kind")
        if kind == "bool_const":
            return BoolConst(bool(data["value"]))
        if kind == "dim_compare":
            return DimCompare(
                Dimension.from_data(data["lhs"]),
                Dimension.from_data(data["rhs"]),
                DimCompareOp(str(data["compare"])),
            )
        if kind == "logical_not":
            return LogicalNot(LogicalExpr.from_data(data["operand"]))
        if kind in {"logical_and", "logical_or"}:
            return LogicalOp(
                str(kind).removeprefix("logical_"),
                tuple(LogicalExpr.from_data(value) for value in data["operands"]),
            )
        raise IRSchemaError(f"Unknown logical expression kind {kind!r}.")

    def __and__(self, other: LogicalExpr) -> LogicalExpr:
        return logical_and(self, other)

    def __or__(self, other: LogicalExpr) -> LogicalExpr:
        return logical_or(self, other)

    def __invert__(self) -> LogicalExpr:
        return logical_not(self)


@dataclass(frozen=True)
class BoolConst(LogicalExpr):
    constant: bool

    @property
    def value(self) -> bool | None:
        return self.constant

    def evaluate(self, bindings: Mapping[str, int]) -> bool:
        return self.constant

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> LogicalExpr:
        return self

    def to_data(self) -> dict[str, object]:
        return {"kind": "bool_const", "value": self.constant}

    def __str__(self) -> str:
        return "true" if self.constant else "false"


@dataclass(frozen=True)
class DimCompare(LogicalExpr):
    """Compare two dimensions; the result is a boolean."""

    lhs: Dimension
    rhs: Dimension
    compare: DimCompareOp = DimCompareOp.EQUAL

    def __post_init__(self) -> None:
        object.__setattr__(self, "lhs", dim(self.lhs))
        object.__setattr__(self, "rhs", dim(self.rhs))
        object.__setattr__(self, "compare", DimCompareOp(self.compare))

    @property
    def value(self) -> bool | None:
        lhs, rhs = simplify_dim(self.lhs), simplify_dim(self.rhs)
        if lhs.is_fixed and rhs.is_fixed:
            return _COMPARISONS[self.compare][1](lhs.fixed_value, rhs.fixed_value)
        # Equality of two provably identical expressions is decidable even
        # when neither side is a constant.
        if equivalent_dim(lhs, rhs):
            return _COMPARISONS[self.compare][1](0, 0)
        return None

    def evaluate(self, bindings: Mapping[str, int]) -> bool:
        return _COMPARISONS[self.compare][1](
            evaluate_dim(self.lhs, bindings), evaluate_dim(self.rhs, bindings))

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> LogicalExpr:
        # Binding every symbol makes the predicate decidable, so fold it.
        return DimCompare(
            substitute_dim(self.lhs, bindings),
            substitute_dim(self.rhs, bindings),
            self.compare,
        ).simplify()

    def simplify(self) -> LogicalExpr:
        decided = self.value
        if decided is not None:
            return BoolConst(decided)
        return DimCompare(simplify_dim(self.lhs), simplify_dim(self.rhs), self.compare)

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "dim_compare",
            "lhs": self.lhs.to_data(),
            "rhs": self.rhs.to_data(),
            "compare": self.compare.value,
        }

    def __str__(self) -> str:
        return f"({self.lhs} {_COMPARISONS[self.compare][0]} {self.rhs})"


@dataclass(frozen=True)
class LogicalNot(LogicalExpr):
    operand: LogicalExpr

    @property
    def value(self) -> bool | None:
        decided = self.operand.value
        return None if decided is None else not decided

    def evaluate(self, bindings: Mapping[str, int]) -> bool:
        return not self.operand.evaluate(bindings)

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> LogicalExpr:
        return LogicalNot(self.operand.substitute(bindings)).simplify()

    def simplify(self) -> LogicalExpr:
        decided = self.value
        if decided is not None:
            return BoolConst(decided)
        return LogicalNot(self.operand.simplify())

    def to_data(self) -> dict[str, object]:
        return {"kind": "logical_not", "operand": self.operand.to_data()}

    def __str__(self) -> str:
        return f"(not {self.operand})"


@dataclass(frozen=True)
class LogicalOp(LogicalExpr):
    """Conjunction or disjunction of two or more predicates."""

    op: str
    operands: tuple[LogicalExpr, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))
        if self.op not in {"and", "or"} or len(self.operands) < 2:
            raise IRSchemaError("A logical operation requires 'and'/'or' and two operands.")
        if any(not isinstance(value, LogicalExpr) for value in self.operands):
            raise IRSchemaError("A logical operation takes boolean expressions.")

    @property
    def value(self) -> bool | None:
        decided = [value.value for value in self.operands]
        if self.op == "and":
            if any(value is False for value in decided):
                return False
            return True if all(value is True for value in decided) else None
        if any(value is True for value in decided):
            return True
        return False if all(value is False for value in decided) else None

    def evaluate(self, bindings: Mapping[str, int]) -> bool:
        values = (value.evaluate(bindings) for value in self.operands)
        return all(values) if self.op == "and" else any(values)

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> LogicalExpr:
        return LogicalOp(
            self.op, tuple(value.substitute(bindings) for value in self.operands)
        ).simplify()

    def simplify(self) -> LogicalExpr:
        decided = self.value
        if decided is not None:
            return BoolConst(decided)
        return LogicalOp(self.op, tuple(value.simplify() for value in self.operands))

    def to_data(self) -> dict[str, object]:
        return {
            "kind": f"logical_{self.op}",
            "operands": [value.to_data() for value in self.operands],
        }

    def __str__(self) -> str:
        symbol = " and " if self.op == "and" else " or "
        return "(" + symbol.join(map(str, self.operands)) + ")"


def dim_compare(
    lhs: DimensionLike,
    rhs: DimensionLike,
    compare: DimCompareOp | str = DimCompareOp.EQUAL,
) -> LogicalExpr:
    """Compare two dimensions, folding the result when it is provable."""
    return DimCompare(dim(lhs), dim(rhs), DimCompareOp(compare)).simplify()


def dim_equal(lhs: DimensionLike, rhs: DimensionLike) -> LogicalExpr:
    return dim_compare(lhs, rhs, DimCompareOp.EQUAL)


def logical_and(*operands: LogicalExpr) -> LogicalExpr:
    return LogicalOp("and", operands).simplify() if len(operands) > 1 else operands[0]


def logical_or(*operands: LogicalExpr) -> LogicalExpr:
    return LogicalOp("or", operands).simplify() if len(operands) > 1 else operands[0]


def logical_not(operand: LogicalExpr) -> LogicalExpr:
    return LogicalNot(operand).simplify()


def logical_const(value: bool) -> LogicalExpr:
    return BoolConst(bool(value))


def logical_from_data(data: Mapping[str, Any]) -> LogicalExpr:
    return LogicalExpr.from_data(data)


__all__ = [
    "BoolConst",
    "DimCompare",
    "LogicalExpr",
    "LogicalNot",
    "LogicalOp",
    "dim_compare",
    "dim_equal",
    "logical_and",
    "logical_const",
    "logical_from_data",
    "logical_not",
    "logical_or",
]
