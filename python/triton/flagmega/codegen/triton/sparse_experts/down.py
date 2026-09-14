# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Local down projections with explicit router-ordered accumulation."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset, stage_context
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.ntt.sparse_experts import SparseExpertsDownCombine
from triton.flagmega.ir.types import data_type, VectorType
from triton.flagmega.codegen.triton.physical_access import emit_triton_scalar_type
from triton.flagmega.codegen.triton.sparse_experts.pipeline import pipeline_context


def sparse_experts_down_call(raw):
    fused = raw["semantic_op"] == SparseExpertsDownCombine.op_name
    context, operands, result = stage_context(raw, SparseExpertsDownCombine if fused else SparseExpertsDown, "down_weight")
    dtype = data_type(raw["semantic_attrs"]["output_dtype"]) if fused and raw["semantic_attrs"]["output_dtype"] is not None else data_type(result["abi"]["scalar_dtype"])
    dtype = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    return {
        **context,
        **pipeline_context(raw, context, operands, ("down",)),
        "combine_routes": fused,
        "weighted_dtype": emit_triton_scalar_type(dtype.value),
        "activation_offset":
        last_axis_offset(operands["activations"]["abi"], ("_fm_token", "_fm_route"), "_fm_k"),
        "weight_offset":
        emit_local_scalar_offset(operands["down_weight"]["abi"], ("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")),
        "probability_offset":
        emit_local_scalar_offset(operands["router_expert_weights"]["abi"], ("_fm_token", "_fm_route")) if fused else None,
        "result_offset":
        last_axis_offset(result["abi"], ("_fm_token",) if fused else ("_fm_token", "_fm_route"), "_fm_n"),
    }
