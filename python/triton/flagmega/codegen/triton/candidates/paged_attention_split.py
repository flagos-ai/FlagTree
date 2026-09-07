# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral semantic TIR for explicit split paged attention."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node

from .core import TritonCandidateContext, TritonCandidateProposal


class PagedAttentionSplitSemanticTIRCandidateProvider:
    """Preserve partial/combine semantics until target implementation choice."""

    op_names = frozenset({
        "ntt.paged_attention_partial",
        "ntt.paged_attention_combine",
    })

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext | None,
    ) -> TritonCandidateProposal | None:
        del context
        if node.op not in self.op_names:
            return None
        candidate = Candidate(f"semantic.{node.op}", {}, {})
        return TritonCandidateProposal(
            (candidate,), candidate.id, selection_kind="semantic_tir"
        )


__all__ = ["PagedAttentionSplitSemanticTIRCandidateProvider"]
