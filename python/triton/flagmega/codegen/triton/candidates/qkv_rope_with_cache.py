# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-neutral semantic TIR proposal for fused Q/K norm, RoPE, and cache IO."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node

from .core import TritonCandidateContext, TritonCandidateProposal


class QKVRoPEWithCacheSemanticTIRCandidateProvider:
    """Preserve the fused operator until target microkernel selection."""

    op_names = frozenset({"nn.qkv_rope_with_cache"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext | None,
    ) -> TritonCandidateProposal | None:
        del context
        if node.op not in self.op_names:
            return None
        candidate = Candidate("semantic.nn.qkv_rope_with_cache", {}, {})
        return TritonCandidateProposal(
            (candidate,), candidate.id, selection_kind="semantic_tir"
        )


__all__ = ["QKVRoPEWithCacheSemanticTIRCandidateProvider"]
