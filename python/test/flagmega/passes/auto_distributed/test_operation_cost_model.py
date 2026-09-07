# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed.operation_cost import (
    DistributedOperationCostModel,
)


def _h800_model():
    return DistributedOperationCostModel(
        block_local_read_bytes_per_cycle=1024,
        block_local_write_bytes_per_cycle=1024,
        block_local_latency_cycles=20,
        elementwise_elements_per_cycle=128,
        simt_fma_per_cycle=64,
        chip_global_read_bytes_per_cycle=1908,
        chip_global_write_bytes_per_cycle=1908,
        chip_global_latency_cycles=300,
        block_synchronization_cycles=25,
        grid_synchronization_cycles=2200,
        identity="test.h800-target-op-cost/v1",
    )


def test_hierarchical_target_cost_matches_nncase_norm_apply_pick_values():
    model = _h800_model()
    placement = fm.Placement((8, 16), "yx", "bb")
    broadcast = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 256)),
        (fm.SBP.broadcast(), fm.SBP.broadcast()),
        placement,
    )
    split = fm.DistributedType(
        broadcast.tensor,
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 2)),
        placement,
    )

    broadcast_factors = fm.OpCostFactors(
        cpu_cycles=1280,
        block_local_memory_load_bytes=12292,
        block_local_memory_store_bytes=4096,
    )
    split_factors = fm.OpCostFactors(
        cpu_cycles=10,
        block_local_memory_load_bytes=100,
        block_local_memory_store_bytes=32,
    )

    assert model.get_latency(broadcast_factors, broadcast) == 1400
    assert model.get_latency(split_factors, split) == 309


def test_target_cost_overlaps_compute_and_memory_then_adds_synchronization():
    model = _h800_model()
    result_type = fm.DistributedType(
        fm.tensor_type("float32", (1,)),
        (fm.SBP.broadcast(),),
        fm.Placement((8, 16), "yx", "bb"),
    )
    factors = fm.OpCostFactors(
        cpu_cycles=2000,
        block_local_memory_load_bytes=1024,
        block_local_memory_store_bytes=1024,
        block_synchronizations=2,
        grid_synchronizations=1,
    )

    # Compute and the two memory paths overlap; synchronization is serialized.
    assert model.get_latency(factors, result_type) == 4250


def test_non_distributed_result_uses_one_active_block():
    model = _h800_model()
    factors = fm.OpCostFactors(block_local_memory_load_bytes=1908)

    assert model.get_latency(factors, fm.tensor_type("float32", (1,))) == 301


def test_hierarchical_target_cost_matches_nncase_packed_matmul_pick_value():
    model = _h800_model()
    result_type = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 2)),
        (fm.SBP.broadcast(), fm.SBP.split_block_cyclic((0, 1), 2)),
        fm.Placement((8, 16), "yx", "bb"),
    )
    factors = fm.OpCostFactors(
        # Padded SIMT GEMV work is target-scaled, while the vector addend's
        # two outer elements are already expressed as evaluator CPU cycles.
        cpu_cycles=2,
        simt_fma_operations=65_536,
        block_local_memory_load_bytes=69_664,
        block_local_memory_store_bytes=32,
    )

    assert model.get_latency(factors, result_type) == 4976
