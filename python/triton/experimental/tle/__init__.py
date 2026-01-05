# Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
# flagtree tle
"""
TLE (Tensor Language Extension) Module

TLE is a Triton language extension for advanced memory management and pipeline operations.
It provides functionality for:
- Local memory allocation (shared memory, tensor memory)
- Bidirectional data copying between global and local memory
- Pipeline-based computation patterns
- Advanced layout encoding and memory management
"""

# Import TLE functionality from the nvidia subdirectory
from .nvidia import (
    # Core operations
    pipeline,
    alloc,
    copy,
    local_load,
    local_store,

    # Types and encodings
    layout,
    shared_layout,
    swizzled_shared_layout,
    tensor_memory_layout,
    nv_mma_shared_layout,
    scope,
    smem,
    tmem,
    buffered_tensor,
    buffered_tensor_type,
)

__all__ = [
    # Core operations
    "pipeline",
    "alloc",
    "copy",
    "local_load",
    "local_store",

    # Types and encodings
    "layout",
    "shared_layout",
    "swizzled_shared_layout",
    "tensor_memory_layout",
    "nv_mma_shared_layout",
    "scope",
    "smem",
    "tmem",
    "buffered_tensor",
    "buffered_tensor_type",
]
