# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.codegen.triton.microkernels import (
    AttentionPrimitiveMicroKernelProvider,
    PackedQKVMicroKernelProvider,
    TIRMicroKernelProviderRegistry,
)
from triton.flagmega.errors import CodegenError


def test_registry_has_one_reviewed_owner_per_semantic_tir_op():
    registry = TIRMicroKernelProviderRegistry()
    registry.add(PackedQKVMicroKernelProvider())

    assert registry.op_names == frozenset({
        "ntt.packed_qkv_parallel_linear_fused_rhs"
    })
    assert isinstance(
        registry.provider_for("ntt.packed_qkv_parallel_linear_fused_rhs"),
        PackedQKVMicroKernelProvider,
    )

    with pytest.raises(CodegenError, match="duplicates semantic op ownership"):
        registry.add(PackedQKVMicroKernelProvider())


def test_default_attention_provider_owns_only_decomposed_semantic_ops():
    registry = TIRMicroKernelProviderRegistry()
    registry.add(AttentionPrimitiveMicroKernelProvider())

    assert registry.op_names == frozenset({
        "nn.rotary_embedding",
        "nn.rope",
            "nn.update_paged_attention_kv_cache",
            "ntt.vectorized_rope",
        })
