# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Compile and evaluate the restricted expression language used by FlagTune YAML.

Operator and model YAML use literals, declared symbol references, and explicit
operation mappings such as ``{op: "mul", args: [M, N]}``.  This module checks
that every reference and operation is declared, then turns the raw YAML into a
small immutable tree.  :mod:`registry` uses the compiled form for input rules,
guards, and ordered model features; it supplies the operation allowlist.

The language intentionally does not parse or execute Python source, perform
attribute access, or import names.  It is not a general expression language:
all available operations and their argument semantics are decided by the
caller, and evaluating an unvalidated raw expression remains unsafe for
untrusted mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

Operation = Callable[..., Any]


class SafeExpressionError(ValueError):
    """Report an invalid symbol name, reference, or safe expression."""


@dataclass(frozen=True)
class Literal:
    """Compiled scalar literal; evaluation returns ``value`` unchanged."""
    value: Any


@dataclass(frozen=True)
class SymbolRef:
    """Compiled reference to one previously declared context symbol."""
    name: str


@dataclass(frozen=True)
class CallExpr:
    """Compiled allowlisted operation call with recursively compiled arguments."""
    op: str
    args: tuple["CompiledExpression", ...]


CompiledExpression = Literal | SymbolRef | CallExpr


def require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or raise a location-aware schema error."""
    if not isinstance(value, Mapping):
        raise SafeExpressionError(f"{location} must be a mapping")
    return value


def require_name(value: Any, location: str) -> str:
    """Return a trimmed non-empty string used by a YAML schema field."""
    if not isinstance(value, str) or not value.strip():
        raise SafeExpressionError(f"{location} must be a non-empty string")
    return value.strip()


def validate_symbol_name(value: Any, location: str) -> str:
    """Validate a YAML symbol shared by fields, tensors, inputs, and features."""
    name = require_name(value, location)
    if not name.isidentifier():
        raise SafeExpressionError(f"{location} must be a valid identifier: {name!r}")
    return name


def _expression_args(expr: Mapping[str, Any], location: str) -> Sequence[Any]:
    args = expr.get("args", [])
    if not isinstance(args, list):
        raise SafeExpressionError(f"{location}.args must be a list")
    return args


def compile_expression(
    expr: Any,
    *,
    symbols: set[str],
    operations: Mapping[str, Operation],
    location: str,
    allow_calls: bool = True,
    allow_literals: bool = True,
) -> CompiledExpression:
    """Compile one YAML value under a location-specific capability set.

    Strings must name a member of ``symbols``.  Mapping calls may use only an
    operation in ``operations`` and the keys ``name``, ``op``, and ``args``;
    numeric, boolean, and null literals are optional.  The returned tree can
    be reused without reparsing YAML.  ``name`` is currently accepted for the
    surrounding feature schema but is not interpreted by this evaluator.
    """
    if isinstance(expr, Mapping):
        if not allow_calls:
            raise SafeExpressionError(f"{location} does not allow operation expressions")
        unknown = set(expr) - {"name", "op", "args"}
        if unknown:
            raise SafeExpressionError(f"{location} has unknown keys: {sorted(unknown)}")
        op_name = validate_symbol_name(expr.get("op"), f"{location}.op")
        if op_name not in operations:
            raise SafeExpressionError(f"{location} uses unknown operation {op_name!r}")
        return CallExpr(
            op_name,
            tuple(
                compile_expression(
                    arg,
                    symbols=symbols,
                    operations=operations,
                    location=f"{location}.args[{index}]",
                    allow_calls=allow_calls,
                    allow_literals=allow_literals,
                ) for index, arg in enumerate(_expression_args(expr, location))),
        )
    if isinstance(expr, str):
        if expr not in symbols:
            raise SafeExpressionError(f"{location} references unknown symbol {expr!r}")
        return SymbolRef(expr)
    if allow_literals and (isinstance(expr, (int, float, bool)) or expr is None):
        return Literal(expr)
    raise SafeExpressionError(f"{location} has unsupported expression value {expr!r}")


def compile_reference(value: Any, *, symbols: set[str], location: str) -> SymbolRef:
    """Compile exactly one reference while rejecting literals and calls."""
    compiled = compile_expression(
        value,
        symbols=symbols,
        operations={},
        location=location,
        allow_calls=False,
        allow_literals=False,
    )
    assert isinstance(compiled, SymbolRef)
    return compiled


def evaluate_compiled(
    expr: CompiledExpression,
    context: Mapping[str, Any],
    operations: Mapping[str, Operation],
) -> Any:
    """Evaluate a compiled tree using the supplied runtime context and operations."""
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, SymbolRef):
        return context[expr.name]
    return operations[expr.op](*(evaluate_compiled(arg, context, operations) for arg in expr.args))


def validate_expression(
    expr: Any,
    operations: Mapping[str, Operation],
    variables: set[str],
    location: str,
) -> None:
    """Validate raw YAML expression syntax without retaining the compiled tree."""
    compile_expression(
        expr,
        symbols=variables,
        operations=operations,
        location=location,
    )


def evaluate_expression(
    expr: Any,
    context: Mapping[str, Any],
    operations: Mapping[str, Operation],
) -> Any:
    """Evaluate an already validated raw expression without executing Python source."""
    if isinstance(expr, Mapping):
        return operations[str(expr["op"])](*(evaluate_expression(arg, context, operations)
                                             for arg in expr.get("args", [])))
    if isinstance(expr, str):
        return context[expr]
    return expr
