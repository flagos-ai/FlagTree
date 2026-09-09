# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Local down projections with explicit router-ordered accumulation."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset, stage_context
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown


def sparse_experts_down_call(raw):
    context, operands, result = stage_context(raw, SparseExpertsDown, "down_weight")
    return {
        **context,
        "activation_offset":
        last_axis_offset(operands["activations"]["abi"], ("_fm_token", "_fm_route"), "_fm_k"),
        "weight_offset":
        emit_local_scalar_offset(operands["down_weight"]["abi"], ("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")),
        "probability_offset":
        emit_local_scalar_offset(operands["router_expert_weights"]["abi"], ("_fm_token", "_fm_route")),
        "result_offset":
        last_axis_offset(result["abi"], ("_fm_token", ), "_fm_n"),
    }
