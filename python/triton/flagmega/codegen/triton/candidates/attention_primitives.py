# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent semantic TIR proposals for decomposed attention ops."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node

from .core import TritonCandidateContext, TritonCandidateProposal


class AttentionPrimitiveSemanticTIRCandidateProvider:
    """Keep attention decomposition semantic until microkernel selection.

    The graph has already been schema-checked by each operation definition.
    This provider intentionally contributes no physical schedule parameters;
    targets may implement the same semantic operations with different kernel
    families without changing importer, rewrite, or pass behavior.
    """

    op_names = frozenset({
        "nn.rotary_embedding",
        "nn.rope",
        "nn.update_paged_attention_kv_cache",
        "ntt.vectorized_rope",
    })

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        del context
        if node.op not in self.op_names:
            return None
        candidate = Candidate(f"semantic.{node.op}", {}, {})
        return TritonCandidateProposal(
            (candidate,), candidate.id, selection_kind="semantic_tir"
        )


__all__ = ["AttentionPrimitiveSemanticTIRCandidateProvider"]
