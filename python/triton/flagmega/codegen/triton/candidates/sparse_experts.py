# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit local expert stages with preserved per-route numerical boundaries."""

from .core import TritonCandidateProposal


class SparseExpertsCandidateProvider:
    op_names = frozenset({"nn.sparse_experts_gate_up", "nn.sparse_experts_down"})

    def propose(self, node, context):
        family = node.op.removeprefix("nn.")
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local", rounding="explicit"))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.simt"))
