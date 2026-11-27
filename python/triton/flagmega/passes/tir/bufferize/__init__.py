# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical bufferization analyses and transformations."""

from triton.flagmega.passes.tir.bufferize.sat_allocator import (
    SATAllocationResult,
    SATBufferAllocator,
)
from triton.flagmega.passes.tir.bufferize.planner import (
    BufferPlanner,
    BufferizationOptions,
    plan_buffers,
)
from triton.flagmega.passes.tir.bufferize.alias_analysis import (
    AliasAnalysis,
    AliasKind,
    PhysicalView,
)
from triton.flagmega.passes.tir.bufferize.synchronization import plan_memory_synchronization
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy
from triton.flagmega.passes.tir.bufferize.first_fit_allocator import FirstFitBufferAllocator
from triton.flagmega.passes.tir.bufferize.allocation import AllocationResult, BufferAllocator, BufferLifetime

__all__ = [
    "BufferLifetime",
    "BufferAllocator",
    "AllocationResult",
    "FirstFitBufferAllocator",
    "BufferPlanner",
    "BufferizationOptions",
    "AliasAnalysis",
    "AliasKind",
    "PhysicalView",
    "NttBufferizationPolicy",
    "SATAllocationResult",
    "SATBufferAllocator",
    "plan_buffers",
    "plan_memory_synchronization",
]
