# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Semantic TIR proposal for partial-statistics normalization apply."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node

from .core import TritonCandidateContext, TritonCandidateProposal


class GatherReduceNormApplySemanticTIRCandidateProvider:
    op_names = frozenset({"ntt.gather_reduce_norm_apply"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext | None,
    ) -> TritonCandidateProposal | None:
        del context
        if node.op not in self.op_names:
            return None
        candidate = Candidate("semantic.ntt.gather_reduce_norm_apply", {}, {})
        return TritonCandidateProposal(
            (candidate,), candidate.id, selection_kind="semantic_tir"
        )


__all__ = ["GatherReduceNormApplySemanticTIRCandidateProvider"]
