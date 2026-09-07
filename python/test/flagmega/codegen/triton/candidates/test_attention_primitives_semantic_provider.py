# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import inspect
from pathlib import Path

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    AttentionPrimitiveSemanticTIRCandidateProvider,
)


@pytest.mark.parametrize(
    ("op", "result_type", "expected"),
    (
        (
            "nn.rotary_embedding",
            fm.TupleType((
                fm.tensor_type("float32", (1, 1, 128)),
                fm.tensor_type("float32", (1, 1, 128)),
            )),
            "semantic.nn.rotary_embedding",
        ),
        ("nn.rope", fm.tensor_type("bfloat16", (1, 16, 128)), "semantic.nn.rope"),
        (
            "ntt.vectorized_rope",
            fm.tensor_type(fm.vector_type("bfloat16", (8,)), (1, 16, 16)),
            "semantic.ntt.vectorized_rope",
        ),
        (
            "nn.update_paged_attention_kv_cache",
            fm.RefType("paged_attention_kv_cache"),
            "semantic.nn.update_paged_attention_kv_cache",
        ),
    ),
)
def test_attention_primitive_proposals_only_select_semantic_tir(
    op, result_type, expected
):
    node = fm.Node("operation", op, (), result_type)

    proposal = AttentionPrimitiveSemanticTIRCandidateProvider().propose(node, None)

    assert proposal is not None
    assert proposal.selection_kind == "semantic_tir"
    assert tuple(value.id for value in proposal.candidates) == (expected,)
    assert proposal.candidates[0].parameters == {}


def test_attention_semantic_provider_contains_no_target_or_schedule_policy():
    source = Path(
        inspect.getsourcefile(AttentionPrimitiveSemanticTIRCandidateProvider) or ""
    ).read_text(encoding="utf-8").lower()

    for spelling in (
        "implementation_model",
        "context.implementations",
        "nvidia",
        "sm90",
        "block_n",
        "block_k",
        "num_warps",
        "num_stages",
    ):
        assert spelling not in source
