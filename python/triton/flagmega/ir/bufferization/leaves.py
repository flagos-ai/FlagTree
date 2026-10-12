# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared naming contract for the physical buffer leaves of a logical value.

Bufferization gives every tensor leaf of an aggregate value its own physical
buffer and its own identity. Planning, alignment propagation and PrimFunction
binding all have to agree on those names, so the flattening order and the
derived identity live here rather than in each consumer.
"""

from __future__ import annotations

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.model import (
    DistributedType,
    IRType,
    NoneType,
    RefType,
    TensorType,
    TupleType,
)


def buffer_leaf_names(prefix: str, value_type: IRType) -> tuple[str, ...]:
    """Buffer identity of each tensor leaf, in physical binding order.

    A single-tensor value owns ``prefix`` itself, so a caller that keys state
    by value identity keeps working unchanged for the non-aggregate case.
    """

    return tuple(name for name, _, _ in buffer_leaves(prefix, value_type))


def buffer_leaves(prefix: str, value_type: IRType):
    """Each ``(buffer id, leaf type, field name)`` of a logical value."""

    if isinstance(value_type, NoneType):
        return ()
    if isinstance(value_type, (TensorType, DistributedType)):
        return ((prefix, value_type, None),)
    if isinstance(value_type, RefType):
        result = []
        for name, field_type in value_type.fields:
            result.extend(buffer_leaves(f"{prefix}.{name}", field_type))
        return tuple(result)
    if isinstance(value_type, TupleType):
        result = []
        for index, field_type in enumerate(value_type.fields):
            result.extend(buffer_leaves(f"{prefix}.{index}", field_type))
        return tuple(result)
    raise IRVerificationError(f"Cannot bufferize {type(value_type).__name__}.")


__all__ = ["buffer_leaf_names", "buffer_leaves"]
