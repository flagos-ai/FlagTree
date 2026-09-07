# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Physical bufferization analyses and transformations."""

from triton.flagmega.passes.tir.bufferize.sat_allocator import (
    BufferLifetime,
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

__all__ = [
    "BufferLifetime",
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
