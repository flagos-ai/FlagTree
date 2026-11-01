# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Indexed expert-bank transfer implementations for SM90."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import T, tensor_type

from .shared_layout import align_shared_workspaces


def sparse_experts_pipeline_implementations():
    result = []
    for family, block_n, fields, sources in (
        ("sparse_experts_gate_up", 64, ("gate", "up"), (3, 6)),
        ("sparse_experts_down", 16, ("down",), (3,)),
    ):
        for dtype, block_k in (("bfloat16", 64), ("float32", 32)):
            workspaces = tuple(T.shared_workspace_descriptor(
                field + "_stage", tensor_type(dtype, (2, 1, block_n, block_k)),
                128, matrix_compatible=True) for field in fields)
            result.append(TritonImplementation(
                id=f"tir.{family}.simt_tma_pipeline_{dtype}", family=family, variant="simt_tma_pipeline",
                parameters={"block_n": block_n, "block_k": block_k, "num_stages": 2,
                            "consumer_warps": 8, "producer_warps": 1, "producer_registers": 24,
                            "compute_num_warps": 8},
                contract={"indexing": "local", "rounding": "explicit", "required_weight_dtype": dtype,
                          "requires_affine_weight_tiles": True, "weight_alignment_bytes": 16,
                          "max_weight_axis_extent": 2**31 - 1},
                requires=("tma", "warp_specialize"),
                facts={"host_tensor_descriptor": True, "transfer_pipeline": True},
                shared_workspaces=align_shared_workspaces(workspaces),
                transfer_pipeline=T.transfer_pipeline_contract(
                    (T.transfer_pipeline_channel("weight", sources, tuple(range(len(fields))), 16),),
                    capacity=2, producer_read_argument_indices=(1,)),
            ))
    return tuple(result)


def sparse_experts_wgmma_implementations():
    return tuple(TritonImplementation(
        f"tir.{family}.wgmma", family, "wgmma",
        {"block_n": 64, "block_k": 128, "compute_num_warps": 4},
        {"indexing": "local", "rounding": "explicit", "requires_block_fp8_wgmma": True},
        requires=("mma_v3", "fp8"),
    ) for family in ("sparse_experts_gate_up", "partitioned_sparse_experts_down"))


def sparse_experts_cooperative_implementations():
    """Optional ordered scale-block cooperation; targets opt in after measurement."""
    return (TritonImplementation(
        "tir.sparse_experts_gate_up.cooperative_wgmma",
        "sparse_experts_gate_up", "cooperative_wgmma",
        {"block_n": 64, "block_k": 128, "compute_num_warps": 8, "merge_n": 32},
        {"indexing": "local", "rounding": "explicit",
         "requires_block_fp8_wgmma": True, "requires_chip_scale_blocks": True},
        requires=("mma_v3", "fp8", "cooperative_grid", "grid_sync"),
        facts={"portable_triton": False, "internal_grid_barriers": 1,
               "source_memory_scope": "chip"},
    ),)


def sparse_experts_complete_k_implementations():
    """Optional full-K tasks publish disjoint rows without global K scratch."""
    return tuple(TritonImplementation(
        f"tir.sparse_experts_gate_up.complete_k_wgmma_n{task_n}",
        "sparse_experts_gate_up", "complete_k_wgmma",
        {"block_n": 64, "block_k": 128, "compute_num_warps": 8, "task_n": task_n, "num_stages": 1},
        {"indexing": "local", "rounding": "explicit",
         "requires_block_fp8_wgmma": True, "requires_chip_scale_blocks": True},
        requires=("mma_v3", "fp8", "cooperative_grid", "grid_sync"),
        facts={"portable_triton": False, "internal_grid_barriers": 1,
               "source_memory_scope": "chip", "result_memory_scope": "chip"},
    ) for task_n in (32, 64))
