# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.microkernels import (
    PagedAttentionSplitMicroKernelProvider,
    default_triton_microkernel_registry,
)


def test_split_attention_microkernel_provider_is_target_neutral_and_registered():
    provider = PagedAttentionSplitMicroKernelProvider()
    registry = default_triton_microkernel_registry()

    assert provider.op_names == frozenset({
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
    })
    assert isinstance(
        registry.provider_for("ntt.paged_attention_partial"),
        PagedAttentionSplitMicroKernelProvider,
    )
    assert isinstance(
        registry.provider_for("ntt.paged_attention_combine"),
        PagedAttentionSplitMicroKernelProvider,
    )
