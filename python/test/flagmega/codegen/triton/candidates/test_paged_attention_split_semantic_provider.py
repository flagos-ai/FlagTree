# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    PagedAttentionSplitSemanticTIRCandidateProvider,
    default_triton_candidate_registry,
)


def test_split_attention_ops_have_target_neutral_semantic_tir_candidates():
    provider = PagedAttentionSplitSemanticTIRCandidateProvider()

    assert provider.op_names == frozenset({
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
    })
    assert provider.propose(
        fm.Node(
            "partial",
            "ntt.paged_attention_partial",
            (),
            fm.TupleType((fm.tensor_type("float32", (1, 1, 1)),)),
        ),
        None,
    ).selection_kind == "semantic_tir"
    assert provider.propose(
        fm.Node(
            "combine",
            "ntt.paged_attention_combine",
            (),
            fm.tensor_type("bfloat16", (1, 1, 1)),
        ),
        None,
    ).selection_kind == "semantic_tir"


def test_default_semantic_tir_catalog_owns_split_attention_ops():
    registry = default_triton_candidate_registry()

    assert isinstance(
        registry.provider_for("ntt.paged_attention_partial"),
        PagedAttentionSplitSemanticTIRCandidateProvider,
    )
    assert isinstance(
        registry.provider_for("ntt.paged_attention_combine"),
        PagedAttentionSplitSemanticTIRCandidateProvider,
    )
