# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Gate/up projection and SwiGLU over an owner's intermediate features."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset, stage_context
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp


def sparse_experts_gate_up_call(raw):
    context, operands, result = stage_context(raw, SparseExpertsGateUp, "gate_weight")
    return {
        **context,
        "q_offset":
        last_axis_offset(operands["q"]["abi"], ("_fm_token", ), "_fm_k"),
        "gate_offset":
        emit_local_scalar_offset(operands["gate_weight"]["abi"], ("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")),
        "up_offset":
        emit_local_scalar_offset(operands["up_weight"]["abi"], ("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")),
        "result_offset":
        last_axis_offset(result["abi"], ("_fm_token", "_fm_route"), "_fm_n"),
    }
