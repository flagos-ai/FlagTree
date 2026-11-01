# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reuse the physical projection proofs for both actual GLU operands."""

from triton.flagmega.codegen.triton.block_fp8_cooperative_renderer import cooperative_block_context
from triton.flagmega.codegen.triton.matmul_glu_geometry import cooperative_glu_geometry
from triton.flagmega.codegen.triton.tensor_transform_renderers import _tensor_type
from triton.flagmega.errors import CodegenError


def cooperative_glu_context(raw, context, operands, result):
    if (
        cooperative_glu_geometry(
            {name: _tensor_type(value["abi"]) for name, value in operands.items()},
            _tensor_type(result["abi"]),
            raw["semantic_attrs"],
        )
        is None
    ):
        raise CodegenError("Cooperative FP8 GLU requires matching complete chip owner domains and SiLU.")
    projections = tuple(
        cooperative_block_context(
            raw,
            context,
            {
                "value": operands["value"],
                "weight": operands[f"{name}_weight"],
                "weight_scale": operands[f"{name}_scale"],
            },
            result,
            partial_name=f"{name}_partials",
            owner="Cooperative FP8 GLU",
        )
        for name in ("gate", "up")
    )
    gate, up = projections
    return {
        **gate,
        "n_mask": f"local_n_offsets < ({context['active_n']})",
        "gate_projection_pointer": gate["projection_pointer"],
        "up_projection_pointer": up["projection_pointer"],
        "up_weight_peer_pointer": up["weight_peer_pointer"],
        "up_weight_peer_offset": up["weight_peer_offset"],
        "up_peer_rows_active": up["peer_rows_active"],
        "up_merge_scale_offset": up["merge_scale_offset"],
    }
