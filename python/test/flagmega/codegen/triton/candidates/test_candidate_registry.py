# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateProposal,
    TritonCandidateProviderRegistry,
)
from triton.flagmega.errors import CodegenError


class _UnitProvider:
    op_names = frozenset({"unit.identity"})

    def propose(self, node, context):
        del node, context
        return TritonCandidateProposal((fm.Candidate(
            "tir.unit.identity",
            {"family": "unit", "variant": "identity"},
            {"portable_triton": True},
        ),), "tir.unit.identity")


def test_registry_rejects_duplicate_semantic_op_ownership():
    registry = TritonCandidateProviderRegistry()
    registry.add(_UnitProvider())

    with pytest.raises(CodegenError, match="duplicates reviewed op ownership"):
        registry.add(_UnitProvider())


def test_proposal_rejects_missing_default_and_duplicate_candidate_ids():
    candidate = fm.Candidate(
        "tir.unit.identity",
        {"family": "unit", "variant": "identity"},
        {},
    )

    with pytest.raises(CodegenError, match="default candidate.*is not in"):
        TritonCandidateProposal((candidate,), "tir.unit.missing")
    with pytest.raises(CodegenError, match="duplicate ids"):
        TritonCandidateProposal((candidate, candidate), candidate.id)
