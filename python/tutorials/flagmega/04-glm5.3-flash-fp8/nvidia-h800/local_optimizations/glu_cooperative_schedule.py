# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Opt-in ordered dense/shared gate-up cooperation with proven physical ABIs."""

from dataclasses import replace
from math import prod

from triton.flagmega.codegen.triton.matmul_glu_geometry import cooperative_glu_geometry
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.matmul_glu import matmul_glu_cooperative_implementations

IMPLEMENTATION = "tir.matmul_glu.cooperative_wgmma"
POLICY = "glm53-ordered-fp8-glu-block-cooperation/v1"


def implementation_model(base):
    (implementation,) = matmul_glu_cooperative_implementations()
    existing = base.implementation(IMPLEMENTATION)
    if existing is not None:
        if existing != implementation:
            raise IRVerificationError("The cooperative GLU implementation has a different contract.")
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
            node.op not in {"nn.matmul_glu", "nn.packed_matmul_glu"}
            or record is None
            or record.candidate_id != "tir.matmul_glu.mma"
        ):
            continue
        geometry = cooperative_glu_geometry(
            {
                parameter.name: module.node_map[node.inputs[index]].type
                for index, parameter in enumerate(get_definition(node.op).input_parameters)
            },
            node.type,
            node.attrs,
        )
        if geometry is None:
            continue
        blocks = prod(n for n, level in zip(geometry.hierarchy, geometry.levels, strict=True) if level == "b")
        if (
            geometry.k_blocks >= 4
            and geometry.local_rows <= 64
            and geometry.row_tiles * geometry.k_blocks >= max(1, blocks // 2)
        ):
            if IMPLEMENTATION not in {candidate.id for candidate in point.candidates}:
                raise IRVerificationError(f"{point.id}: ordered cooperative GLU is unavailable.")
            choices.append((point.id, IMPLEMENTATION))
    return apply_plan(
        module,
        override_plan(
            module,
            choices,
            policy=POLICY,
            rationale="Measured complete GLU stage; preserve K128 quantization, combined-scale FMA and every BF16 boundary.",
        ),
    )
