# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect
from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    QKVRoPEWithCacheSemanticTIRCandidateProvider,
    default_triton_candidate_registry,
)


def test_qkv_rope_with_cache_is_preserved_as_target_neutral_semantic_tir():
    result_type = fm.TupleType((
        fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 16, 16)),
        fm.RefType("paged_attention_kv_cache"),
    ))
    node = fm.Node("fused", "nn.qkv_rope_with_cache", (), result_type)

    proposal = QKVRoPEWithCacheSemanticTIRCandidateProvider().propose(node, None)

    assert proposal is not None
    assert proposal.selection_kind == "semantic_tir"
    assert tuple(value.id for value in proposal.candidates) == (
        "semantic.nn.qkv_rope_with_cache",
    )
    assert proposal.candidates[0].parameters == {}
    assert isinstance(
        default_triton_candidate_registry().provider_for(
            "nn.qkv_rope_with_cache"
        ),
        QKVRoPEWithCacheSemanticTIRCandidateProvider,
    )


def test_qkv_rope_with_cache_semantic_provider_has_no_model_or_machine_policy():
    source = Path(
        inspect.getsourcefile(QKVRoPEWithCacheSemanticTIRCandidateProvider) or ""
    ).read_text(encoding="utf-8").lower()

    for spelling in (
        "qwen",
        "nvidia",
        "sm90",
        "block_n",
        "block_k",
        "num_warps",
        "num_stages",
    ):
        assert spelling not in source
