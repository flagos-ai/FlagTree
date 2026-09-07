# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.ir.bufferization.alias import AliasInfo, AliasKind
from triton.flagmega.ir.bufferization.descriptor import BufferDescriptor
from triton.flagmega.ir.bufferization.function_plan import (
    CallBufferBinding,
    FunctionBufferPlan,
    KernelCallBufferBinding,
)
from triton.flagmega.ir.bufferization.mem_span import MemSpan
from triton.flagmega.ir.bufferization.memory_pool import (
    CallMemoryPoolBinding,
    FunctionMemoryPool,
)
from triton.flagmega.ir.bufferization.memory import (
    AllocationPolicy,
    AllocationStrategy,
    MemoryAllocationScope,
    MemorySharingScope,
    MemorySpace,
)
from triton.flagmega.ir.bufferization.physical_buffer import PhysicalAllocation, PhysicalBuffer
from triton.flagmega.ir.bufferization.plan import (
    BUFFER_PLAN_SCHEMA,
    LEGACY_BUFFER_PLAN_SCHEMA,
    BufferPlan,
)
from triton.flagmega.ir.bufferization.synchronization import (
    MemoryRange,
    MemorySynchronizationPlan,
    SYNCHRONIZATION_SCHEMA,
    SynchronizationEvent,
)

__all__ = [
    "AllocationPolicy",
    "AllocationStrategy",
    "AliasInfo",
    "AliasKind",
    "BUFFER_PLAN_SCHEMA",
    "BufferDescriptor",
    "BufferPlan",
    "CallBufferBinding",
    "CallMemoryPoolBinding",
    "FunctionBufferPlan",
    "FunctionMemoryPool",
    "KernelCallBufferBinding",
    "LEGACY_BUFFER_PLAN_SCHEMA",
    "MemoryAllocationScope",
    "MemorySharingScope",
    "MemorySpace",
    "MemSpan",
    "MemoryRange",
    "MemorySynchronizationPlan",
    "PhysicalAllocation",
    "PhysicalBuffer",
    "SYNCHRONIZATION_SCHEMA",
    "SynchronizationEvent",
]
