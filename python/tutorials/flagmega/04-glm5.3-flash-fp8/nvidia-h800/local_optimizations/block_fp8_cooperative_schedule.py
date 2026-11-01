# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in measured ordered FP8 projection cooperation for the decode profile."""

from dataclasses import replace
from math import prod

from triton.flagmega.codegen.triton.block_fp8_geometry import cooperative_block_geometry
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.block_fp8 import block_fp8_cooperative_implementations

IMPLEMENTATION = "tir.block_fp8.cooperative_wgmma"
POLICY = "glm53-ordered-fp8-projection-block-cooperation/v1"


def implementation_model(base):
    (implementation,) = block_fp8_cooperative_implementations()
    existing = base.implementation(IMPLEMENTATION)
    if existing is not None:
        if existing != implementation:
            raise IRVerificationError("The cooperative FP8 projection has a different registered contract.")
        return base
    return replace(base, implementations=(*base.implementations, implementation))


def select(module):
    choices = []
    for point in module.selection_points:
        if point.kind != "tir" or point.owner not in module.node_map:
            continue
        node = module.node_map[point.owner]
        record = module.selection_map.get(point.id)
        if (
            node.op not in {"math.block_scaled_matmul", "math.packed_block_scaled_matmul"}
            or record is None
            or record.candidate_id != "tir.block_fp8.mma"
        ):
            continue
        geometry = cooperative_block_geometry(
            dict(
                zip(
                    ("value", "weight", "weight_scale"),
                    (module.node_map[value].type for value in node.inputs),
                    strict=True,
                )
            ),
            node.type,
            node.attrs,
        )
        if geometry is None:
            continue
        blocks = prod(n for n, level in zip(geometry.hierarchy, geometry.levels, strict=True) if level == "b")
        # Four-block projections still remove eightfold padded-row work in
        # the measured N8 domain; half-grid raw occupancy pays for its publish.
        if (
            geometry.k_blocks >= 4
            and geometry.local_rows <= 64
            and geometry.row_tiles * geometry.k_blocks >= max(1, blocks // 2)
        ):
            if IMPLEMENTATION not in {candidate.id for candidate in point.candidates}:
                raise IRVerificationError(f"{point.id}: ordered cooperative FP8 projection unavailable.")
            choices.append((point.id, IMPLEMENTATION))
    return apply_plan(
        module,
        override_plan(
            module,
            choices,
            policy=POLICY,
            rationale="Measured independent K128 raw dot blocks; preserve combined-scale multiplication and ordered FMA.",
        ),
    )
