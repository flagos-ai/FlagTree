# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Local down projections with explicit router-ordered accumulation."""

from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset, emit_triton_scalar_type
from triton.flagmega.codegen.triton.sparse_experts.common import last_axis_offset, stage_context
from triton.flagmega.codegen.triton.sparse_experts.pipeline import pipeline_context
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.ops.nn.partitioned_sparse_experts_down import PartitionedSparseExpertsDown
from triton.flagmega.ir.ops.nn.sparse_experts_down import SparseExpertsDown
from triton.flagmega.ir.ops.ntt.sparse_experts import SparseExpertsDownCombine
from triton.flagmega.ir.types import VectorType, data_type


def sparse_experts_down_call(raw):
    fused = raw["semantic_op"] == SparseExpertsDownCombine.op_name
    partitioned = raw["semantic_op"] == PartitionedSparseExpertsDown.op_name
    definition = (PartitionedSparseExpertsDown if partitioned else
                  SparseExpertsDownCombine if fused else SparseExpertsDown)
    output_tile_n = raw["parameters"].get("output_tile_n", 64)
    if "output_tile_n" in raw["parameters"] and (not partitioned or raw["variant"] != "wgmma"):
        raise CodegenError("output_tile_n requires the partitioned FP8 WGMMA down implementation.")
    if (isinstance(output_tile_n, bool) or not isinstance(output_tile_n, int)
            or output_tile_n < 64 or output_tile_n & (output_tile_n - 1)):
        raise CodegenError("output_tile_n must be a power of two of at least 64.")
    context, operands, result = stage_context(raw, definition, "down_weight", partitioned=partitioned)
    tasks = raw["parameters"].get("tasks_per_owner", 1)
    if (type(tasks) is not int or tasks < 1
            or tasks > 1 and (not partitioned or raw["variant"] != "wgmma"
                              or context["scalar_n"] != tasks * output_tile_n)):
        raise CodegenError("Down tasks must exactly partition each owner's output into WGMMA tiles.")
    dtype = data_type(raw["semantic_attrs"]["output_dtype"]) if fused and raw["semantic_attrs"]["output_dtype"] is not None else data_type(result["abi"]["scalar_dtype"])
    dtype = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    return {
        **context,
        **pipeline_context(raw, context, operands, ("down",)),
        "combine_routes": fused,
        "output_tile_n": output_tile_n,
        "tasks_per_owner": tasks,
        **({"task_partition": {"tasks_per_owner": tasks, "owner_extent": context["scalar_n"],
                                "task_extent": output_tile_n, "disjoint_writes": True}} if tasks > 1 else {}),
        "weighted_dtype": emit_triton_scalar_type(dtype.value),
        "activation_offset":
        last_axis_offset(operands["activations"]["abi"],
                         ("_fm_token", "_fm_route", "_fm_partition") if partitioned else ("_fm_token", "_fm_route"), "_fm_k"),
        "weight_offset":
        emit_local_scalar_offset(operands["down_weight"]["abi"],
            ("_fm_expert", "_fm_partition", "_fm_n[:, None]", "_fm_k[None, :]") if partitioned
            else ("_fm_expert", "_fm_n[:, None]", "_fm_k[None, :]")),
        "probability_offset":
        emit_local_scalar_offset(operands["router_expert_weights"]["abi"], ("_fm_token", "_fm_route")) if fused else None,
        "result_offset":
        last_axis_offset(result["abi"], ("_fm_partition", "_fm_token", "_fm_route") if partitioned else
                         ("_fm_token",) if fused else ("_fm_token", "_fm_route"), "_fm_n"),
    }
