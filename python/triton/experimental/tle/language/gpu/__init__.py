# flagtree tle
from .core import (
    pipeline,
    alloc,
    copy,
    memory_space,
    local_ptr,
)
from .types import (layout, shared_layout, swizzled_shared_layout, tensor_memory_layout, nv_mma_shared_layout, scope,
                    buffered_tensor, buffered_tensor_type, smem, tmem)

# Backward-compat alias expected by existing tests/tutorials.
storage_kind = memory_space

__all__ = [
    "pipeline",
    "alloc",
    "copy",
    "local_ptr",
    "storage_kind",
    "layout",
    "memory_space",
    "shared_layout",
    "swizzled_shared_layout",
    "tensor_memory_layout",
    "nv_mma_shared_layout",
    "scope",
    "buffered_tensor",
    "buffered_tensor_type",
    "smem",
    "tmem",
]
