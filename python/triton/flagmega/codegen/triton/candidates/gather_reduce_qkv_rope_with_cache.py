# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Semantic TIR proposal for fused partial-QKV reduction and RoPE/cache IO."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node

from .core import TritonCandidateContext, TritonCandidateProposal


class GatherReduceQKVRoPEWithCacheSemanticTIRCandidateProvider:
    """Keep cross-owner reduction explicit until microkernel selection."""

    op_names = frozenset({"ntt.gather_reduce_qkv_rope_with_cache"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext | None,
    ) -> TritonCandidateProposal | None:
        del context
        if node.op not in self.op_names:
            return None
        candidate = Candidate(
            "semantic.ntt.gather_reduce_qkv_rope_with_cache", {}, {}
        )
        return TritonCandidateProposal(
            (candidate,), candidate.id, selection_kind="semantic_tir"
        )


__all__ = ["GatherReduceQKVRoPEWithCacheSemanticTIRCandidateProvider"]
