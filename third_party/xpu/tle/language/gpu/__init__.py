# Triton 3.6 XPU TLE - GPU module
# Phase 1: contiguous-copy APIs
from .core import (
    alloc,
    local_ptr,
    copy,
    dma_wait,
    local_load,
    local_store,
    pipeline,
    memory_space,
)
from .types import (
    scope,
    lmem,
    smem,
    buffered_tensor,
    buffered_tensor_type,
)

__all__ = [
    "alloc",
    "local_ptr",
    "copy",
    "dma_wait",
    "local_load",
    "local_store",
    "pipeline",
    "memory_space",
    "scope",
    "lmem",
    "smem",
    "buffered_tensor",
    "buffered_tensor_type",
]
