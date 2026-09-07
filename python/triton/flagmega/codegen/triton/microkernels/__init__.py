# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral semantic-TIR to target-microkernel selection."""

from .core import (
    TIRMicroKernelContext,
    TIRMicroKernelProposal,
    TIRMicroKernelProvider,
    TIRMicroKernelProviderRegistry,
)
from .packed_qkv import PackedQKVMicroKernelProvider
from .attention_primitives import AttentionPrimitiveMicroKernelProvider
from .paged_attention_split import PagedAttentionSplitMicroKernelProvider
from .qkv_rope_with_cache import QKVRoPEWithCacheMicroKernelProvider
from .gather_reduce_qkv_rope_with_cache import (
    GatherReduceQKVRoPEWithCacheMicroKernelProvider,
)
from .gather_reduce_norm_apply import GatherReduceNormApplyMicroKernelProvider
from .selection import TritonMicroKernelSelectionPolicy


def default_triton_microkernel_registry() -> TIRMicroKernelProviderRegistry:
    registry = TIRMicroKernelProviderRegistry()
    registry.add(PackedQKVMicroKernelProvider())
    registry.add(AttentionPrimitiveMicroKernelProvider())
    registry.add(PagedAttentionSplitMicroKernelProvider())
    registry.add(QKVRoPEWithCacheMicroKernelProvider())
    registry.add(GatherReduceQKVRoPEWithCacheMicroKernelProvider())
    registry.add(GatherReduceNormApplyMicroKernelProvider())
    return registry


__all__ = [
    "AttentionPrimitiveMicroKernelProvider",
    "PackedQKVMicroKernelProvider",
    "PagedAttentionSplitMicroKernelProvider",
    "QKVRoPEWithCacheMicroKernelProvider",
    "GatherReduceQKVRoPEWithCacheMicroKernelProvider",
    "GatherReduceNormApplyMicroKernelProvider",
    "TIRMicroKernelContext",
    "TIRMicroKernelProposal",
    "TIRMicroKernelProvider",
    "TIRMicroKernelProviderRegistry",
    "TritonMicroKernelSelectionPolicy",
    "default_triton_microkernel_registry",
]
