# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The two GLU projections must share one complete chip-local row domain."""

from triton.flagmega.codegen.triton.block_fp8_geometry import cooperative_block_geometry
from triton.flagmega.ir import DType, logical_type


def cooperative_glu_geometry(types, result_type, attrs):
    if (
        attrs.get("activation") != "silu"
        or attrs.get("gate_weight_encoding") is not None
        or attrs.get("up_weight_encoding") is not None
        or logical_type(result_type).dtype is not DType.BFLOAT16
    ):
        return None
    geometries = tuple(
        cooperative_block_geometry(
            {"value": types["value"], "weight": types[f"{name}_weight"], "weight_scale": types[f"{name}_scale"]},
            result_type,
            attrs,
        )
        for name in ("gate", "up")
    )
    return geometries[0] if geometries[0] is not None and geometries[0] == geometries[1] else None
