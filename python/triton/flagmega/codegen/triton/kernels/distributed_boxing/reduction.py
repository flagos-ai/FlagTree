# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed neutral elements and associative operations for owner reductions."""

import re

from triton.flagmega.errors import CodegenError


def partial_reduction_context(reduce_op: str, dtype: str) -> dict[str, str]:
    if reduce_op not in {"sum", "min", "max", "prod"}:
        raise CodegenError(f"Unsupported Partial Boxing reduction {reduce_op!r}.")
    integer = re.fullmatch(r"(u?)int(8|16|32|64)", dtype)
    if dtype in {"float16", "bfloat16", "float32"}:
        accumulator = "tl.float32"
        lower, upper = "float('-inf')", "float('inf')"
    elif integer:
        unsigned = bool(integer[1])
        bits = int(integer[2])
        accumulator = f"tl.{'u' if unsigned else ''}int{max(32, bits)}"
        lower = "0" if unsigned else str(-(1 << (bits - 1)))
        upper = str((1 << (bits if unsigned else bits - 1)) - 1)
    else:
        raise CodegenError(f"Partial Boxing has no reviewed reduction dtype {dtype!r}.")
    neutral = {"sum": "0", "prod": "1", "min": upper, "max": lower}[reduce_op]
    reduction = (
        "tl.reduce(boxing_partial_values, axis=0, combine_fn=_flagmega_boxing_multiply)"
        if reduce_op == "prod" else f"tl.{reduce_op}(boxing_partial_values, axis=0)"
    )
    combine = {
        "sum": "boxing_accumulator + boxing_reduced",
        "prod": "boxing_accumulator * boxing_reduced",
        "min": "tl.minimum(boxing_accumulator, boxing_reduced)",
        "max": "tl.maximum(boxing_accumulator, boxing_reduced)",
    }[reduce_op]
    return {
        "mode": f"partial_{reduce_op}",
        "accumulator_type": accumulator,
        "neutral": neutral,
        "reduction_expression": reduction,
        "combine_expression": combine,
    }
