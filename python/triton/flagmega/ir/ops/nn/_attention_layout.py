# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Shared semantic-axis helpers for attention operations."""

from __future__ import annotations

from triton.flagmega.errors import EvaluationError, IRSchemaError


ATTENTION_AXES = ("seq", "head", "dim")


def normalize_attention_layout(value: object) -> tuple[str, str, str]:
    if not isinstance(value, (tuple, list)):
        raise IRSchemaError("Attention layout must be a sequence of semantic axes.")
    layout = tuple(str(axis).lower() for axis in value)
    if len(layout) != 3 or set(layout) != set(ATTENTION_AXES):
        raise IRSchemaError(
            "Attention layout must be a permutation of ('seq', 'head', 'dim').")
    return layout


def to_seq_head_dim(value, layout: tuple[str, str, str]):
    permutation = tuple(layout.index(axis) for axis in ATTENTION_AXES)
    return value if permutation == (0, 1, 2) else value.permute(permutation)


def from_seq_head_dim(value, layout: tuple[str, str, str]):
    permutation = tuple(ATTENTION_AXES.index(axis) for axis in layout)
    return value if permutation == (0, 1, 2) else value.permute(permutation)


def require_decode_token(value, *, operation: str):
    if value.ndim != 3 or value.shape[0] != 1:
        raise EvaluationError(
            f"{operation} reference evaluator requires one [seq, head, dim] decode token.")
    return value[0]


__all__ = [
    "ATTENTION_AXES",
    "from_seq_head_dim",
    "normalize_attention_layout",
    "require_decode_token",
    "to_seq_head_dim",
]
