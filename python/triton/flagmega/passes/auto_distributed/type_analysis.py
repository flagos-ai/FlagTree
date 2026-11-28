# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Memoized type-only op queries; no weight values or candidate IDs are shared."""

from collections.abc import Mapping

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import Node


def _attribute_key(value):
    if isinstance(value, Mapping):
        return (Mapping, tuple((key, _attribute_key(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (tuple, tuple(_attribute_key(item) for item in value))
    return (type(value), value)


def infer_candidate(context, definition, input_types):
    attrs = context.source_call.attrs
    # Method identity also invalidates an in-place agent override of the op.
    key = (definition.infer_type, definition.cost_factors, tuple(input_types), _attribute_key(attrs))
    memo = context.type_inference_memo
    if key not in memo:
        # These are type-only placeholders, not the original SSA inputs. The
        # generic inference provider has always queried types through Vars.
        inputs = tuple(
            Node(f"<distributed-input.{i}>", "builtin.var", (), value, attrs={"name": f"<distributed-input.{i}>"})
            for i, value in enumerate(input_types))
        try:
            result_type = definition.infer_type(inputs, attrs)
        except (IRSchemaError, AssertionError, ValueError):
            memo[key] = None
        else:
            memo[key] = (result_type, definition.cost_factors(inputs, attrs, result_type))
    return memo[key]


__all__ = ["infer_candidate"]
