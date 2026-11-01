# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tiled-output variants of the owner-local PackedQKV transfer contract."""

from dataclasses import replace

from triton.flagmega.ir import T, tensor_type
from .shared_layout import align_shared_workspaces


def packed_qkv_n_tiled_implementations(partial_mma):
    result = []
    contract = dict(partial_mma.contract)
    del contract["required_local_output_extent"]
    for kind in ("gemv", "mma"):
        for descriptor in ("single", "table"):
            block_n, block_k, stages = 64, 64, 2
            shape = ((1,) if descriptor == "single" else ()) + (block_n // 8, block_k // 16, 2, 64)
            workspaces = (T.shared_workspace_descriptor(
                "rhs_stage", tensor_type("bfloat16", (stages, *shape)), 128, matrix_compatible=True),
                partial_mma.shared_workspaces[1])
            parameters = {**partial_mma.parameters, "block_n": block_n, "block_k": block_k,
                          "num_stages": stages, "descriptor_kind": descriptor, "n_tiling": True}
            facts = dict(partial_mma.facts)
            if kind == "gemv":
                parameters["reduction_group"] = 32
                facts.pop("matrix_primitive")
            result.append(replace(
                partial_mma,
                id=f"tir.qkv_parallel_linear.packed_partial_{kind}_n_tiled_{descriptor}_pipeline",
                variant=f"packed_{kind}_smem_pipeline", parameters=parameters, contract=contract, facts=facts,
                requires=tuple(value for value in partial_mma.requires if kind == "mma" or value != "mma_v3"),
                shared_workspaces=align_shared_workspaces(workspaces),
                transfer_pipeline=replace(partial_mma.transfer_pipeline, capacity=stages),
            ))
            direct_contract = {key: value for key, value in contract.items() if key not in {
                "required_local_reduction_extent", "requires_uniform_full_input_reduction_tiles"}}
            direct_contract["requires_matching_packed_extents"] = True
            result.append(replace(
                result[-1],
                id=f"tir.qkv_parallel_linear.packed_partial_{kind}_n_tiled_{descriptor}_direct_pipeline",
                parameters={**parameters, "direct_lhs": True}, contract=direct_contract,
                shared_workspaces=align_shared_workspaces(workspaces[:1]),
                transfer_pipeline=replace(result[-1].transfer_pipeline, consumer_shared_workspace_indices=()),
            ))
    return tuple(result)


def packed_qkv_ordered_implementations(partial_mma):
    """Keep the fused transfer pipeline and one accumulator across all K."""
    variants = packed_qkv_n_tiled_implementations(partial_mma)
    base = next(value for value in variants if value.id == (
        "tir.qkv_parallel_linear.packed_partial_mma_n_tiled_table_direct_pipeline"))
    contract = {key: value for key, value in base.contract.items() if key not in {
        "required_output_partial_reduce_op", "requires_partial_axes_match_input_reduction_ownership"}}
    contract.update(reduction_order="mma_k16", required_output_dtype="bfloat16")
    facts = {key: value for key, value in base.facts.items() if key != "produces_sum_partials"}
    pipelined = replace(
        base, id="tir.qkv_parallel_linear.packed_ordered_mma_table_pipeline",
        contract=contract, facts={**facts, "ordered_mma_accumulation": True})
    from triton.flagmega.codegen.triton.implementation import TritonImplementation
    tiled = TritonImplementation(
        "tir.qkv_parallel_linear.packed_ordered_mma", "qkv_parallel_linear", "packed_ordered_mma",
        {"block_k": 128, "tile_n": 16},
        {"input_kind": "fused_rhs", "rhs_layout": "k_major", "reduction_order": "mma_k16",
         "required_input_dtype": "bfloat16", "required_weight_dtype": "bfloat16"},
        requires=("mma_v3",), facts={"ordered_mma_accumulation": True})
    return pipelined, tiled
