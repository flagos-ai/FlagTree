# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Ordered MoE scale-block cooperation for the explicit native decode profile."""

from dataclasses import replace
from math import prod

from triton.flagmega.codegen.triton.sparse_experts.cooperative_geometry import cooperative_gate_geometry
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.sparse_experts import sparse_experts_cooperative_implementations

IMPLEMENTATION = "tir.sparse_experts_gate_up.cooperative_wgmma"
POLICY = "glm53-ordered-moe-scale-block-cooperation/v1"


def implementation_model(base):
    (implementation,) = sparse_experts_cooperative_implementations()
    existing = base.implementation(IMPLEMENTATION)
    if existing is not None:
        if existing != implementation:
            raise IRVerificationError("The cooperative expert implementation has a different registered contract.")
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
            node.op != "nn.sparse_experts_gate_up"
            or record is None
            or record.candidate_id != "tir.sparse_experts_gate_up.wgmma"
        ):
            continue
        geometry = cooperative_gate_geometry(
            {
                parameter.name: module.node_map[node.inputs[index]].type
                for index, parameter in enumerate(get_definition(node.op).input_parameters)
            },
            node.type,
        )
        if geometry is None:
            continue
        blocks = prod(n for n, level in zip(geometry.hierarchy, geometry.levels, strict=True) if level == "b")
        if (
            geometry.k_blocks >= 8
            and geometry.local_rows <= 64
            and geometry.chip_routes * geometry.row_tiles * geometry.k_blocks >= blocks
        ):
            if IMPLEMENTATION not in {value.id for value in point.candidates}:
                raise IRVerificationError(f"{point.id}: ordered cooperative expert candidate unavailable.")
            choices.append((point.id, IMPLEMENTATION))
    return apply_plan(
        module,
        override_plan(
            module,
            choices,
            policy=POLICY,
            rationale="Measured independent FP8 K128 scale blocks; preserve original B multiplication and ordered A FMA.",
        ),
    )
