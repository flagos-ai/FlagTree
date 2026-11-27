# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SM90 memory-space contract for physical bufferization."""

from triton.flagmega.ir.bufferization import (
    AllocationPolicy,
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.passes.tir.bufferize import BufferizationOptions
from triton.flagmega.targets.nvidia.capability import Sm90Capability


def sm90_bufferization_options(capability: Sm90Capability) -> BufferizationOptions:
    maximum = (1 << 62) - 1
    return BufferizationOptions((
        MemorySpace(
            "workspace", "device", 256, maximum, AllocationStrategy.REUSE,
            AllocationPolicy.GRANULARITY_ALIGNED,
            MemoryAllocationScope.FUNCTION, MemorySharingScope.CHIP,
        ),
        MemorySpace(
            "block_local_data", "device", 256, 64 * 1024 * 1024,
            AllocationStrategy.REUSE,
            AllocationPolicy.GRANULARITY_ALIGNED,
            MemoryAllocationScope.FUNCTION, MemorySharingScope.BLOCK,
        ),
        MemorySpace(
            "rdata", "readonly_device", 256, maximum, AllocationStrategy.LINEAR,
            AllocationPolicy.GRANULARITY_ALIGNED,
            MemoryAllocationScope.MODULE, MemorySharingScope.CHIP,
        ),
        MemorySpace(
            "shared", "shared", 16, capability.max_shared_memory_bytes,
            AllocationStrategy.REUSE, AllocationPolicy.GRANULARITY_ALIGNED,
            MemoryAllocationScope.FUNCTION, MemorySharingScope.BLOCK,
        ),
        MemorySpace(
            "external", "external", 1, maximum, AllocationStrategy.EXTERNAL,
            AllocationPolicy.GRANULARITY_ALIGNED,
            MemoryAllocationScope.EXTERNAL, MemorySharingScope.CHIP,
        ),
    ), block_local="block_local_data")


__all__ = ["sm90_bufferization_options"]
