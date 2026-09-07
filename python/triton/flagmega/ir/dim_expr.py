# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable integer dimension expressions with SymPy-backed simplification.

The public tree is owned by FlagMega and remains stable in Python IR dumps.
SymPy is used only to canonicalize supported integer algebra and to prove
equivalence; no SymPy object is serialized into compiler checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, reduce
from operator import add, mul
from typing import Any, Iterable, Mapping

from triton.flagmega.errors import IRSchemaError


class DimensionKind(str, Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"
    UNKNOWN = "unknown"


class DimCompareOp(str, Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_OR_EQUAL = "less_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_OR_EQUAL = "greater_or_equal"


class Dimension:
    """Base class for fixed, dynamic, unknown and calculated dimensions."""

    @property
    def kind(self) -> DimensionKind:
        raise NotImplementedError

    @property
    def value(self) -> int | None:
        return None

    @property
    def name(self) -> str | None:
        return None

    @property
    def minimum(self) -> int | None:
        return _bounds(self)[0]

    @property
    def maximum(self) -> int | None:
        return _bounds(self)[1]

    @property
    def is_fixed(self) -> bool:
        return self.kind == DimensionKind.FIXED

    @property
    def is_dynamic(self) -> bool:
        return self.kind == DimensionKind.DYNAMIC

    @property
    def is_unknown(self) -> bool:
        return self.kind == DimensionKind.UNKNOWN

    @property
    def fixed_value(self) -> int:
        if self.value is None:
            raise IRSchemaError(f"Dimension {self} is not fixed.")
        return self.value

    def simplify(self) -> Dimension:
        return simplify_dim(self)

    def evaluate(self, bindings: Mapping[str, int]) -> int:
        return evaluate_dim(self, bindings)

    def substitute(self, bindings: Mapping[str, int | Dimension]) -> Dimension:
        return substitute_dim(self, bindings)

    def equivalent(self, other: int | str | Dimension) -> bool:
        return equivalent_dim(self, dim(other))

    def to_data(self) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def from_data(data: Mapping[str, Any]) -> Dimension:
        kind = data.get("kind")
        if kind == "fixed":
            return DimConst(int(data["value"]))
        if kind in {"symbolic", "variable"}:
            return DimVar(
                str(data["name"]),
                _optional_int(data.get("minimum")),
                _optional_int(data.get("maximum")),
            )
        if kind == "unknown":
            return UNKNOWN_DIM
        if kind == "expr":
            return dim_expr(
                str(data["op"]),
                *(Dimension.from_data(value) for value in data.get("operands", ())),
            )
        raise IRSchemaError(f"Unknown dimension kind {kind!r}.")

    def __add__(self, other: int | str | Dimension) -> Dimension:
        return dim_expr("add", self, dim(other))

    def __radd__(self, other: int | str | Dimension) -> Dimension:
        return dim(other) + self

    def __sub__(self, other: int | str | Dimension) -> Dimension:
        return dim_expr("add", self, -dim(other))

    def __rsub__(self, other: int | str | Dimension) -> Dimension:
        return dim(other) - self

    def __mul__(self, other: int | str | Dimension) -> Dimension:
        return dim_expr("mul", self, dim(other))

    def __rmul__(self, other: int | str | Dimension) -> Dimension:
        return dim(other) * self

    def __floordiv__(self, other: int | str | Dimension) -> Dimension:
        return floor_div(self, other)

    def __rfloordiv__(self, other: int | str | Dimension) -> Dimension:
        return floor_div(other, self)

    def __truediv__(self, other: int | str | Dimension) -> Dimension:
        return floor_div(self, other)

    def __rtruediv__(self, other: int | str | Dimension) -> Dimension:
        return floor_div(other, self)

    def __mod__(self, other: int | str | Dimension) -> Dimension:
        return dim_mod(self, other)

    def __rmod__(self, other: int | str | Dimension) -> Dimension:
        return dim_mod(other, self)

    def __pow__(self, power: int) -> Dimension:
        return dim_pow(self, power)

    def __neg__(self) -> Dimension:
        return dim_expr("mul", DimConst(-1), self)


@dataclass(frozen=True)
class DimConst(Dimension):
    fixed: int

    def __post_init__(self) -> None:
        if isinstance(self.fixed, bool):
            raise IRSchemaError("Boolean is not a valid dimension constant.")
        object.__setattr__(self, "fixed", int(self.fixed))

    @property
    def kind(self) -> DimensionKind:
        return DimensionKind.FIXED

    @property
    def value(self) -> int:
        return self.fixed

    @property
    def minimum(self) -> int:
        return self.fixed

    @property
    def maximum(self) -> int:
        return self.fixed

    def to_data(self) -> dict[str, object]:
        return {"kind": "fixed", "value": self.fixed}

    def __str__(self) -> str:
        return str(self.fixed)


@dataclass(frozen=True)
class DimVar(Dimension):
    symbol: str
    lower_bound: int | None = None
    upper_bound: int | None = None

    def __post_init__(self) -> None:
        if not self.symbol or not self.symbol.isidentifier():
            raise IRSchemaError(f"Dimension variable requires an identifier name, got {self.symbol!r}.")
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise IRSchemaError("A dimension variable minimum cannot exceed its maximum.")

    @property
    def kind(self) -> DimensionKind:
        return DimensionKind.DYNAMIC

    @property
    def name(self) -> str:
        return self.symbol

    @property
    def minimum(self) -> int | None:
        return self.lower_bound

    @property
    def maximum(self) -> int | None:
        return self.upper_bound

    def to_data(self) -> dict[str, object]:
        return {
            # Keep the v1 spelling so existing editable checkpoints retain
            # their semantic hash. ``from_data`` also accepts ``variable``.
            "kind": "symbolic",
            "name": self.symbol,
            "minimum": self.lower_bound,
            "maximum": self.upper_bound,
        }

    def __str__(self) -> str:
        return self.symbol


@dataclass(frozen=True)
class UnknownDim(Dimension):
    @property
    def kind(self) -> DimensionKind:
        return DimensionKind.UNKNOWN

    def to_data(self) -> dict[str, object]:
        return {"kind": "unknown"}

    def __str__(self) -> str:
        return "?"


UNKNOWN_DIM = UnknownDim()


@dataclass(frozen=True)
class DimExpr(Dimension):
    op: str
    operands: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(dim(value) for value in self.operands))
        _validate_expression(self.op, self.operands)

    @property
    def kind(self) -> DimensionKind:
        return DimensionKind.DYNAMIC

    def to_data(self) -> dict[str, object]:
        return {
            "kind": "expr",
            "op": self.op,
            "operands": [operand.to_data() for operand in self.operands],
        }

    def __str__(self) -> str:
        if self.op == "add":
            return "(" + " + ".join(map(str, self.operands)) + ")"
        if self.op == "mul":
            return "(" + " * ".join(map(str, self.operands)) + ")"
        if self.op in {"floor_div", "ceil_div", "mod", "pow"}:
            symbol = {"floor_div": "//", "ceil_div": "ceil_div", "mod": "%", "pow": "**"}[self.op]
            return f"({self.operands[0]} {symbol} {self.operands[1]})"
        if self.op.startswith("select_"):
            symbol = {
                "select_eq": "==",
                "select_ne": "!=",
                "select_lt": "<",
                "select_le": "<=",
                "select_gt": ">",
                "select_ge": ">=",
            }[self.op]
            value, expected, true_value, false_value = self.operands
            return f"({value} {symbol} {expected} ? {true_value} : {false_value})"
        if self.op == "positive":
            return f"positive({self.operands[0]}, {self.operands[1]})"
        return f"{self.op}({', '.join(map(str, self.operands))})"


DimensionLike = int | str | Dimension


def dim(value: DimensionLike, minimum: int | None = None, maximum: int | None = None) -> Dimension:
    if isinstance(value, Dimension):
        if minimum is not None or maximum is not None:
            raise TypeError("Bounds can only be supplied when constructing a named dimension variable.")
        return value
    if isinstance(value, bool):
        raise IRSchemaError("Boolean is not a valid dimension.")
    if isinstance(value, int):
        if minimum is not None or maximum is not None:
            raise TypeError("Fixed dimensions do not accept symbolic bounds.")
        return DimConst(value)
    return DimVar(str(value), minimum, maximum)


def unknown_dim() -> UnknownDim:
    return UNKNOWN_DIM


def dim_expr(op: str, *operands: DimensionLike) -> Dimension:
    values = tuple(dim(value) for value in operands)
    _validate_expression(op, values)
    simplified = _simplify_local(op, values)
    if not isinstance(simplified, DimExpr):
        return simplified
    return _simplify_sympy(simplified)


def dim_add(*operands: DimensionLike) -> Dimension:
    return dim_expr("add", *operands)


def dim_mul(*operands: DimensionLike) -> Dimension:
    return dim_expr("mul", *operands)


def floor_div(lhs: DimensionLike, rhs: DimensionLike) -> Dimension:
    return dim_expr("floor_div", lhs, rhs)


def ceil_div(lhs: DimensionLike, rhs: DimensionLike) -> Dimension:
    return dim_expr("ceil_div", lhs, rhs)


def dim_mod(lhs: DimensionLike, rhs: DimensionLike) -> Dimension:
    return dim_expr("mod", lhs, rhs)


def dim_pow(value: DimensionLike, power: int) -> Dimension:
    if isinstance(power, bool) or not isinstance(power, int):
        raise IRSchemaError("Dimension power must be an integer.")
    return dim_expr("pow", value, power)


def dim_min(*operands: DimensionLike) -> Dimension:
    return dim_expr("min", *operands)


def dim_max(*operands: DimensionLike) -> Dimension:
    return dim_expr("max", *operands)


def dim_abs(value: DimensionLike) -> Dimension:
    return dim_expr("abs", value)


def dim_clamp(value: DimensionLike, minimum: DimensionLike, maximum: DimensionLike) -> Dimension:
    return dim_expr("clamp", value, minimum, maximum)


def dim_select(
    value: DimensionLike,
    expected: DimensionLike,
    true_value: DimensionLike,
    false_value: DimensionLike,
    compare: DimCompareOp | str = DimCompareOp.EQUAL,
) -> Dimension:
    comparison = DimCompareOp(compare)
    op = {
        DimCompareOp.EQUAL: "select_eq",
        DimCompareOp.NOT_EQUAL: "select_ne",
        DimCompareOp.LESS_THAN: "select_lt",
        DimCompareOp.LESS_OR_EQUAL: "select_le",
        DimCompareOp.GREATER_THAN: "select_gt",
        DimCompareOp.GREATER_OR_EQUAL: "select_ge",
    }[comparison]
    return dim_expr(op, value, expected, true_value, false_value)


def dim_positive(value: DimensionLike, extent: DimensionLike) -> Dimension:
    """Normalize a possibly-negative index against an extent."""

    return dim_expr("positive", value, extent)


def align_up(value: DimensionLike, alignment: int) -> Dimension:
    if isinstance(alignment, bool) or alignment <= 0:
        raise IRSchemaError("Dimension alignment must be a positive integer.")
    return ceil_div(value, alignment) * alignment


def simplify_dim(value: DimensionLike) -> Dimension:
    value = dim(value)
    if not isinstance(value, DimExpr):
        return value
    operands = tuple(simplify_dim(operand) for operand in value.operands)
    return dim_expr(value.op, *operands)


def evaluate_dim(value: DimensionLike, bindings: Mapping[str, int]) -> int:
    value = dim(value)
    if isinstance(value, DimConst):
        return value.fixed
    if isinstance(value, DimVar):
        if value.symbol not in bindings:
            raise IRSchemaError(f"No runtime binding was supplied for dimension {value.symbol!r}.")
        result = int(bindings[value.symbol])
        if value.minimum is not None and result < value.minimum:
            raise IRSchemaError(f"Dimension {value.symbol!r} value {result} is below {value.minimum}.")
        if value.maximum is not None and result > value.maximum:
            raise IRSchemaError(f"Dimension {value.symbol!r} value {result} exceeds {value.maximum}.")
        return result
    if isinstance(value, UnknownDim):
        raise IRSchemaError("An unknown dimension cannot be evaluated.")
    values = tuple(evaluate_dim(operand, bindings) for operand in value.operands)
    if value.op == "add":
        return sum(values)
    if value.op == "mul":
        return reduce(mul, values, 1)
    if value.op == "floor_div":
        return values[0] // values[1]
    if value.op == "ceil_div":
        return -(-values[0] // values[1])
    if value.op == "mod":
        return values[0] % values[1]
    if value.op == "pow":
        return values[0] ** values[1]
    if value.op == "min":
        return min(values)
    if value.op == "max":
        return max(values)
    if value.op == "abs":
        return abs(values[0])
    if value.op == "clamp":
        return min(max(values[0], values[1]), values[2])
    if value.op == "positive":
        return values[0] if values[0] >= 0 else values[0] + values[1]
    if value.op.startswith("select_"):
        return values[2] if _compare_values(value.op, values[0], values[1]) else values[3]
    raise IRSchemaError(f"Cannot evaluate dimension op {value.op!r}.")


def substitute_dim(value: DimensionLike, bindings: Mapping[str, int | Dimension]) -> Dimension:
    value = dim(value)
    if isinstance(value, DimVar) and value.symbol in bindings:
        return dim(bindings[value.symbol])
    if isinstance(value, DimExpr):
        return dim_expr(value.op, *(substitute_dim(operand, bindings) for operand in value.operands))
    return value


def equivalent_dim(lhs: DimensionLike, rhs: DimensionLike) -> bool:
    lhs_value = simplify_dim(lhs)
    rhs_value = simplify_dim(rhs)
    if lhs_value == rhs_value:
        return True
    if lhs_value.is_unknown or rhs_value.is_unknown:
        return False
    try:
        sympy, lhs_expr, _ = _to_sympy(lhs_value)
        _, rhs_expr, _ = _to_sympy(rhs_value)
        return sympy.simplify(lhs_expr - rhs_expr) == 0
    except (ImportError, TypeError, ValueError):
        return False


def try_div_exactly(value: DimensionLike, divisor: DimensionLike) -> Dimension | None:
    divisor = dim(divisor)
    if isinstance(divisor, DimConst) and divisor.fixed <= 0:
        raise IRSchemaError("Exact dimension divisor must be positive.")
    value = simplify_dim(value)
    remainder = dim_mod(value, divisor)
    if isinstance(remainder, DimConst) and remainder.fixed == 0:
        return floor_div(value, divisor)
    quotient = floor_div(value, divisor)
    return quotient if equivalent_dim(quotient * divisor, value) else None


def _validate_expression(op: str, operands: tuple[Dimension, ...]) -> None:
    arities = {
        "floor_div": 2,
        "ceil_div": 2,
        "mod": 2,
        "pow": 2,
        "abs": 1,
        "clamp": 3,
        "positive": 2,
        "select_eq": 4,
        "select_ne": 4,
        "select_lt": 4,
        "select_le": 4,
        "select_gt": 4,
        "select_ge": 4,
    }
    if op in {"add", "mul", "min", "max"} and not operands:
        raise IRSchemaError(f"Dimension {op} requires at least one operand.")
    if op in arities and len(operands) != arities[op]:
        raise IRSchemaError(f"Dimension {op} requires {arities[op]} operands, got {len(operands)}.")
    if op not in {
        "add", "mul", "floor_div", "ceil_div", "mod", "pow", "min", "max", "abs", "clamp",
        "positive", "select_eq", "select_ne", "select_lt", "select_le", "select_gt", "select_ge",
    }:
        raise IRSchemaError(f"Unsupported dimension expression op {op!r}.")
    if op in {"floor_div", "ceil_div", "mod"} and isinstance(operands[1], DimConst) and operands[1].fixed == 0:
        raise IRSchemaError(f"Dimension {op} denominator cannot be zero.")
    if op == "pow" and (not isinstance(operands[1], DimConst) or operands[1].fixed < 0):
        raise IRSchemaError("Dimension power currently requires a non-negative fixed exponent.")


def _simplify_local(op: str, operands: tuple[Dimension, ...]) -> Dimension:
    if op == "mul" and any(isinstance(value, DimConst) and value.fixed == 0 for value in operands):
        return DimConst(0)
    if any(value.is_unknown for value in operands):
        return UNKNOWN_DIM
    if all(isinstance(value, DimConst) for value in operands):
        return DimConst(evaluate_dim(DimExpr(op, operands), {}))
    if op in {"add", "mul"}:
        flattened: list[Dimension] = []
        identity = 0 if op == "add" else 1
        constant = identity
        for operand in operands:
            if isinstance(operand, DimExpr) and operand.op == op:
                nested = operand.operands
            else:
                nested = (operand,)
            for value in nested:
                if isinstance(value, DimConst):
                    constant = constant + value.fixed if op == "add" else constant * value.fixed
                else:
                    flattened.append(value)
        if constant != identity or not flattened:
            flattened.append(DimConst(constant))
        flattened = [value for value in flattened if not (isinstance(value, DimConst) and value.fixed == identity)]
        if not flattened:
            return DimConst(identity)
        if len(flattened) == 1:
            return flattened[0]
        return DimExpr(op, tuple(flattened))
    if op in {"min", "max"}:
        unique = tuple(dict.fromkeys(operands))
        if len(unique) == 1:
            return unique[0]
        return DimExpr(op, unique)
    if op == "pow" and isinstance(operands[1], DimConst):
        if operands[1].fixed == 0:
            return DimConst(1)
        if operands[1].fixed == 1:
            return operands[0]
    if op in {"floor_div", "ceil_div"} and operands[1] == DimConst(1):
        return operands[0]
    if op == "mod" and operands[1] == DimConst(1):
        return DimConst(0)
    if op == "abs":
        minimum, _ = _bounds(operands[0])
        if minimum is not None and minimum >= 0:
            return operands[0]
    if op == "clamp" and operands[1] == operands[2]:
        return operands[1]
    if op == "positive":
        minimum, _ = _bounds(operands[0])
        if minimum is not None and minimum >= 0:
            return operands[0]
    if op.startswith("select_"):
        if operands[2] == operands[3]:
            return operands[2]
        condition = _prove_comparison(op, operands[0], operands[1])
        if condition is not None:
            return operands[2] if condition else operands[3]
    return DimExpr(op, operands)


@lru_cache(maxsize=4096)
def _simplify_sympy(value: DimExpr) -> Dimension:
    """Canonicalize an immutable expression once per compiler process.

    Codegen derives the same local-shard coordinate expressions for every
    occurrence of a physical buffer in call ABI, schedule, and runtime-binding
    analyses. SymPy simplification is pure but comparatively expensive, so a
    bounded structural cache prevents those consumers from paying for the same
    proof repeatedly without making SymPy objects part of serialized IR.
    """

    try:
        sympy, expression, variables = _to_sympy(value)
        simplified = sympy.simplify(expression)
        return _from_sympy(sympy, simplified, variables)
    except (ImportError, TypeError, ValueError, NotImplementedError):
        return value


def _to_sympy(value: Dimension):
    try:
        import sympy
    except ImportError as error:
        raise ImportError("DimExpr simplification requires the FlagMega optional dependency 'sympy'.") from error
    variables: dict[str, DimVar] = {}

    def convert(current: Dimension):
        if isinstance(current, DimConst):
            return sympy.Integer(current.fixed)
        if isinstance(current, DimVar):
            existing = variables.get(current.symbol)
            if existing is not None and existing != current:
                raise ValueError(f"Dimension variable {current.symbol!r} has inconsistent bounds.")
            variables[current.symbol] = current
            return sympy.Symbol(current.symbol, integer=True)
        if isinstance(current, UnknownDim):
            raise ValueError("Unknown dimensions cannot be converted to SymPy.")
        values = [convert(operand) for operand in current.operands]
        if current.op == "add":
            return sympy.Add(*values)
        if current.op == "mul":
            return sympy.Mul(*values)
        if current.op == "floor_div":
            return sympy.floor(values[0] / values[1])
        if current.op == "ceil_div":
            return sympy.ceiling(values[0] / values[1])
        if current.op == "mod":
            return sympy.Mod(values[0], values[1])
        if current.op == "pow":
            return sympy.Pow(values[0], values[1])
        if current.op == "min":
            return sympy.Min(*values)
        if current.op == "max":
            return sympy.Max(*values)
        if current.op == "abs":
            return sympy.Abs(values[0])
        if current.op == "clamp":
            return sympy.Min(sympy.Max(values[0], values[1]), values[2])
        if current.op == "positive":
            return sympy.Piecewise((values[0], values[0] >= 0), (values[0] + values[1], True))
        if current.op.startswith("select_"):
            comparison = {
                "select_eq": sympy.Eq,
                "select_ne": sympy.Ne,
                "select_lt": sympy.Lt,
                "select_le": sympy.Le,
                "select_gt": sympy.Gt,
                "select_ge": sympy.Ge,
            }[current.op](values[0], values[1])
            return sympy.Piecewise((values[2], comparison), (values[3], True))
        raise NotImplementedError(current.op)

    return sympy, convert(value), variables


def _from_sympy(sympy, value, variables: Mapping[str, DimVar]) -> Dimension:
    if value.is_Integer:
        return DimConst(int(value))
    if value.is_Symbol:
        return variables.get(str(value), DimVar(str(value)))
    if value.func == sympy.Add:
        return _simplify_local("add", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.Mul:
        return _simplify_local("mul", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.Pow:
        return _simplify_local("pow", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.floor:
        numerator, denominator = sympy.fraction(value.args[0])
        return DimExpr("floor_div", (_from_sympy(sympy, numerator, variables), _from_sympy(sympy, denominator, variables)))
    if value.func == sympy.ceiling:
        numerator, denominator = sympy.fraction(value.args[0])
        return DimExpr("ceil_div", (_from_sympy(sympy, numerator, variables), _from_sympy(sympy, denominator, variables)))
    if value.func == sympy.Mod:
        return DimExpr("mod", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.Min:
        return DimExpr("min", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.Max:
        return DimExpr("max", tuple(_from_sympy(sympy, item, variables) for item in value.args))
    if value.func == sympy.Abs:
        return DimExpr("abs", (_from_sympy(sympy, value.args[0], variables),))
    raise NotImplementedError(f"Unsupported simplified SymPy dimension {value!r}.")


def _bounds(value: Dimension) -> tuple[int | None, int | None]:
    if isinstance(value, DimConst):
        return value.fixed, value.fixed
    if isinstance(value, DimVar):
        return value.lower_bound, value.upper_bound
    if isinstance(value, UnknownDim):
        return None, None
    bounds = tuple(_bounds(operand) for operand in value.operands)
    if value.op == "add":
        return _sum_bound(item[0] for item in bounds), _sum_bound(item[1] for item in bounds)
    if value.op == "mul" and all(low is not None and high is not None for low, high in bounds):
        minimum, maximum = 1, 1
        for low, high in bounds:
            candidates = (minimum * low, minimum * high, maximum * low, maximum * high)  # type: ignore[operator]
            minimum, maximum = min(candidates), max(candidates)
        return minimum, maximum
    if value.op == "mod" and isinstance(value.operands[1], DimConst) and value.operands[1].fixed > 0:
        return 0, value.operands[1].fixed - 1
    if value.op in {"floor_div", "ceil_div"}:
        low, high = bounds[0]
        denominator = value.operands[1]
        if isinstance(denominator, DimConst) and denominator.fixed > 0:
            operation = (lambda item: item // denominator.fixed) if value.op == "floor_div" else (
                lambda item: -(-item // denominator.fixed))
            return None if low is None else operation(low), None if high is None else operation(high)
    if value.op == "abs":
        low, high = bounds[0]
        if low is not None and high is not None:
            candidates = (abs(low), abs(high), 0 if low <= 0 <= high else min(abs(low), abs(high)))
            return min(candidates), max(candidates)
        return 0, None
    if value.op in {"min", "max"}:
        lows = [item[0] for item in bounds]
        highs = [item[1] for item in bounds]
        function = min if value.op == "min" else max
        return (
            None if any(item is None for item in lows) else function(lows),
            None if any(item is None for item in highs) else function(highs),
        )
    if value.op == "clamp":
        return bounds[1][0], bounds[2][1]
    if value.op == "positive":
        low, high = bounds[0]
        _, extent_high = bounds[1]
        return 0 if low is None or low < 0 else low, extent_high if high is None else max(high, extent_high or high)
    if value.op.startswith("select_"):
        true_low, true_high = bounds[2]
        false_low, false_high = bounds[3]
        return _optional_min(true_low, false_low), _optional_max(true_high, false_high)
    return None, None


def _sum_bound(values: Iterable[int | None]) -> int | None:
    values = tuple(values)
    return None if any(value is None for value in values) else sum(values)  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _compare_values(op: str, lhs: int, rhs: int) -> bool:
    return {
        "select_eq": lhs == rhs,
        "select_ne": lhs != rhs,
        "select_lt": lhs < rhs,
        "select_le": lhs <= rhs,
        "select_gt": lhs > rhs,
        "select_ge": lhs >= rhs,
    }[op]


def _prove_comparison(op: str, lhs: Dimension, rhs: Dimension) -> bool | None:
    if lhs == rhs:
        return op in {"select_eq", "select_le", "select_ge"}
    if isinstance(lhs, DimConst) and isinstance(rhs, DimConst):
        return _compare_values(op, lhs.fixed, rhs.fixed)
    lhs_low, lhs_high = _bounds(lhs)
    rhs_low, rhs_high = _bounds(rhs)
    if op == "select_eq":
        if lhs_high is not None and rhs_low is not None and lhs_high < rhs_low:
            return False
        if rhs_high is not None and lhs_low is not None and rhs_high < lhs_low:
            return False
    if op == "select_ne":
        equal = _prove_comparison("select_eq", lhs, rhs)
        return None if equal is None else not equal
    proofs = {
        "select_lt": (
            lhs_high is not None and rhs_low is not None and lhs_high < rhs_low,
            lhs_low is not None and rhs_high is not None and lhs_low >= rhs_high,
        ),
        "select_le": (
            lhs_high is not None and rhs_low is not None and lhs_high <= rhs_low,
            lhs_low is not None and rhs_high is not None and lhs_low > rhs_high,
        ),
        "select_gt": (
            lhs_low is not None and rhs_high is not None and lhs_low > rhs_high,
            lhs_high is not None and rhs_low is not None and lhs_high <= rhs_low,
        ),
        "select_ge": (
            lhs_low is not None and rhs_high is not None and lhs_low >= rhs_high,
            lhs_high is not None and rhs_low is not None and lhs_high < rhs_low,
        ),
    }
    if op in proofs:
        proven_true, proven_false = proofs[op]
        return True if proven_true else False if proven_false else None
    return None


def _optional_min(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else min(lhs, rhs)


def _optional_max(lhs: int | None, rhs: int | None) -> int | None:
    return None if lhs is None or rhs is None else max(lhs, rhs)


__all__ = [
    "DimConst",
    "DimCompareOp",
    "DimExpr",
    "DimVar",
    "Dimension",
    "DimensionKind",
    "UnknownDim",
    "align_up",
    "ceil_div",
    "dim",
    "dim_abs",
    "dim_add",
    "dim_clamp",
    "dim_expr",
    "dim_max",
    "dim_min",
    "dim_mod",
    "dim_mul",
    "dim_pow",
    "dim_positive",
    "dim_select",
    "equivalent_dim",
    "evaluate_dim",
    "floor_div",
    "simplify_dim",
    "substitute_dim",
    "try_div_exactly",
    "unknown_dim",
]
