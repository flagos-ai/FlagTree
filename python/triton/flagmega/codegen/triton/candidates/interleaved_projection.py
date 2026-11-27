# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native row projection with two producer-owned result fields."""

from .core import TritonCandidateContext, TritonCandidateProposal


class InterleavedProjectionCandidateProvider:
    op_names = frozenset({"nn.interleaved_projection"})

    def propose(self, node, context: TritonCandidateContext):
        del node
        candidates = tuple(context.configure_implementation(implementation)
                           for implementation in context.implementations("interleaved_projection"))
        if not candidates:
            return None
        return TritonCandidateProposal(candidates, context.choose_default("interleaved_projection", candidates))
