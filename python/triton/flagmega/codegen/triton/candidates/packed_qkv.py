# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Semantic TIR proposal for portable Packed-QKV dataflow."""

from __future__ import annotations

from triton.flagmega.ir import Candidate, Node, TupleType

from .core import TritonCandidateContext, TritonCandidateProposal


class PackedQKVSemanticTIRCandidateProvider:
    """Select semantic Packed-QKV TIR without consulting target implementations."""

    op_names = frozenset({"ntt.packed_qkv_parallel_linear"})

    def propose(
        self,
        node: Node,
        context: TritonCandidateContext,
    ) -> TritonCandidateProposal | None:
        del context
        if (
            node.attrs.get("rhs_layout") != "k_major"
            or not isinstance(node.type, TupleType)
            or len(node.type.fields) != 3
        ):
            return None
        candidate = Candidate(
            "semantic.ntt.packed_qkv_parallel_linear",
            {},
            {"canonicalize_fused_rhs_before_microkernel_selection": True},
        )
        return TritonCandidateProposal(
            (candidate,),
            candidate.id,
            selection_kind="semantic_tir",
        )


__all__ = ["PackedQKVSemanticTIRCandidateProvider"]
