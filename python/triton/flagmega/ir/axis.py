# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Axis validation independent of operation registration order."""

from triton.flagmega.errors import IRSchemaError


def normalize_axis(axis: int, rank: int) -> int:
    if isinstance(axis, bool) or not isinstance(axis, int) or not -rank <= axis < rank:
        raise IRSchemaError(f"Axis {axis!r} is out of range for rank {rank}.")
    return axis % rank
