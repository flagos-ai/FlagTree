# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Hierarchical target aggregation for AutoDistribution operation factors."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

from triton.flagmega.ir import DistributedType, IRType, OpCostFactors, TupleType


@dataclass(frozen=True)
class DistributedOperationCostModel:
    """Map op-local factors to target cycles without owning graph policy.

    This is the Python counterpart of nncase's ``TritonTargetOpCostModel``.
    Block-local work is evaluated both against the nearest block memory and
    against aggregate chip-global traffic across every active placement
    owner. Compute and memory paths overlap; synchronization is serialized.

    The defaults form a deterministic unit machine for isolated provider
    tests. Production targets inject their physical machine profile.
    """

    block_local_read_bytes_per_cycle: int = 1
    block_local_write_bytes_per_cycle: int = 1
    block_local_latency_cycles: int = 0
    elementwise_elements_per_cycle: int = 1
    simt_fma_per_cycle: int = 1
    chip_global_read_bytes_per_cycle: int = 1
    chip_global_write_bytes_per_cycle: int = 1
    chip_global_latency_cycles: int = 0
    block_synchronization_cycles: int = 1
    grid_synchronization_cycles: int = 1
    identity: str = "flagmega.target-op-cost.unit/v1"

    def __post_init__(self) -> None:
        rates = (
            self.block_local_read_bytes_per_cycle,
            self.block_local_write_bytes_per_cycle,
            self.elementwise_elements_per_cycle,
            self.simt_fma_per_cycle,
            self.chip_global_read_bytes_per_cycle,
            self.chip_global_write_bytes_per_cycle,
        )
        latencies = (
            self.block_local_latency_cycles,
            self.chip_global_latency_cycles,
            self.block_synchronization_cycles,
            self.grid_synchronization_cycles,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in rates):
            raise ValueError("Distributed operation bandwidths must be positive integers.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in latencies):
            raise ValueError("Distributed operation latencies must be non-negative integers.")
        if not self.identity:
            raise ValueError("DistributedOperationCostModel requires an identity.")

    def get_latency(self, factors: OpCostFactors, result_type: IRType) -> int:
        active_blocks = _active_block_count(result_type)
        local_load = factors.block_local_memory_load_bytes
        local_store = factors.block_local_memory_store_bytes
        chip_load = factors.chip_global_memory_load_bytes
        chip_store = factors.chip_global_memory_store_bytes

        block_memory_cycles = (
            local_load / self.block_local_read_bytes_per_cycle
            + local_store / self.block_local_write_bytes_per_cycle
            + (
                self.block_local_latency_cycles
                if local_load + local_store > 0
                else 0
            )
        )
        chip_memory_bytes = local_load + local_store + chip_load + chip_store
        chip_memory_cycles = (
            ((local_load + chip_load) * active_blocks)
            / self.chip_global_read_bytes_per_cycle
            + ((local_store + chip_store) * active_blocks)
            / self.chip_global_write_bytes_per_cycle
            + (self.chip_global_latency_cycles if chip_memory_bytes > 0 else 0)
        )
        compute_cycles = (
            factors.cpu_cycles
            + ceil(
                factors.elementwise_operations
                / self.elementwise_elements_per_cycle
            )
            + ceil(factors.simt_fma_operations / self.simt_fma_per_cycle)
        )
        overlapped_cycles = max(
            compute_cycles,
            block_memory_cycles,
            chip_memory_cycles,
        )
        latency = (
            overlapped_cycles
            + factors.block_synchronizations
            * self.block_synchronization_cycles
            + factors.grid_synchronizations
            * self.grid_synchronization_cycles
            + factors.communication_cycles
        )
        return min(max(ceil(latency), 0), 2_000_000_000)


def _active_block_count(value: IRType) -> int:
    if isinstance(value, DistributedType):
        return max(prod(value.placement.hierarchy), 1)
    if isinstance(value, TupleType):
        return max((_active_block_count(field) for field in value.fields), default=1)
    return 1


__all__ = ["DistributedOperationCostModel"]
