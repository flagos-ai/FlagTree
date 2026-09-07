# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Composable type predicates used by op parameters and graph patterns.

This is the Python counterpart of nncase ``TypePattern``.  A pattern owns both
the predicate and a stable human-readable reason, so construction, verifier,
pattern matching and diagnostics share one type contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.model import (
    AnyType,
    CallableType,
    DistributedType,
    IRType,
    InvalidType,
    NoneType,
    RefType,
    TensorType,
    TupleType,
)
from triton.flagmega.ir.types import MaskVectorType, PointerType, VectorType, DataType, data_type


TypeCondition = Callable[[IRType], bool]


@dataclass(frozen=True)
class TypePattern:
    """A composable predicate over :class:`IRType` with a diagnostic reason."""

    condition: TypeCondition = field(compare=False, repr=False)
    reason: str

    def match_leaf(self, value_type: IRType) -> bool:
        return isinstance(value_type, IRType) and bool(self.condition(value_type))

    def check(self, value_type: IRType, field_name: str) -> IRType:
        if not self.match_leaf(value_type):
            raise IRSchemaError(
                f"Parameter {field_name!r} requires <{self.reason}>, "
                f"but got {type(value_type).__name__}: {value_type!r}."
            )
        return value_type

    def __and__(self, other: TypePattern) -> TypePattern:
        if not isinstance(other, TypePattern):
            return NotImplemented
        return TypePattern(
            lambda value: self.match_leaf(value) and other.match_leaf(value),
            f"<{self.reason}> and <{other.reason}>",
        )

    def __or__(self, other: TypePattern) -> TypePattern:
        if not isinstance(other, TypePattern):
            return NotImplemented
        return TypePattern(
            lambda value: self.match_leaf(value) or other.match_leaf(value),
            f"<{self.reason}> or <{other.reason}>",
        )

    def __invert__(self) -> TypePattern:
        return TypePattern(lambda value: not self.match_leaf(value), f"not <{self.reason}>")


def is_type(value_or_condition: IRType | TypeCondition, reason: str | None = None) -> TypePattern:
    if isinstance(value_or_condition, IRType):
        expected = value_or_condition
        return TypePattern(lambda value: value == expected, reason or f"type = {expected!r}")
    if not callable(value_or_condition):
        raise TypeError("is_type expects an IRType or callable condition.")
    return TypePattern(value_or_condition, reason or getattr(value_or_condition, "__name__", "custom type condition"))


def is_ir_type() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, IRType), "is_ir_type")


def is_tensor() -> TypePattern:
    return TypePattern(
        lambda value: isinstance(value, TensorType)
        or isinstance(value, DistributedType) and isinstance(value.tensor, TensorType),
        "is_tensor",
    )


def is_any() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, AnyType), "is_any")


def is_callable() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, CallableType), "is_callable")


def is_invalid() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, InvalidType), "is_invalid")


def is_none() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, NoneType), "is_none")


def is_distributed() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, DistributedType), "is_distributed")


def is_tuple() -> TypePattern:
    return TypePattern(lambda value: isinstance(value, TupleType), "is_tuple")


def is_ref(name: str | None = None) -> TypePattern:
    reason = "is_ref" if name is None else f"is_ref({name})"
    return TypePattern(lambda value: isinstance(value, RefType) and (name is None or value.name == name), reason)


def is_pointer() -> TypePattern:
    return TypePattern(
        lambda value: isinstance(value, TensorType)
        and value.rank == 0
        and isinstance(value.dtype, PointerType),
        "is_pointer",
    )


def is_vector() -> TypePattern:
    return TypePattern(
        lambda value: isinstance(value, TensorType) and isinstance(value.dtype, VectorType),
        "is_vector",
    )


def is_mask_vector() -> TypePattern:
    return TypePattern(
        lambda value: isinstance(value, TensorType) and isinstance(value.dtype, MaskVectorType),
        "is_mask_vector",
    )


def has_rank(rank: int) -> TypePattern:
    if isinstance(rank, bool) or rank < 0:
        raise ValueError(f"Rank must be a non-negative integer, got {rank!r}.")
    return TypePattern(
        lambda value: (
            value.tensor.rank == rank if isinstance(value, DistributedType) else
            isinstance(value, TensorType) and value.rank == rank
        ),
        f"rank = {rank}",
    )


def has_dtype(dtype: DataType | str) -> TypePattern:
    expected = data_type(dtype)
    return TypePattern(
        lambda value: (
            value.tensor.dtype == expected if isinstance(value, DistributedType) else
            isinstance(value, TensorType) and value.dtype == expected
        ),
        f"dtype = {expected.value}",
    )


def tuple_fields(count: int) -> TypePattern:
    if isinstance(count, bool) or count < 0:
        raise ValueError(f"Tuple field count must be a non-negative integer, got {count!r}.")
    return TypePattern(
        lambda value: isinstance(value, TupleType) and len(value.fields) == count,
        f"tuple fields = {count}",
    )


__all__ = [
    "TypePattern",
    "has_dtype",
    "has_rank",
    "is_any",
    "is_callable",
    "is_distributed",
    "is_ir_type",
    "is_invalid",
    "is_none",
    "is_pointer",
    "is_ref",
    "is_mask_vector",
    "is_tensor",
    "is_tuple",
    "is_vector",
    "is_type",
    "tuple_fields",
]
