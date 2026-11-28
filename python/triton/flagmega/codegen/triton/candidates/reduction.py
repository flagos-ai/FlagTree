# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-local reductions with explicit semantic axes and tiled storage access."""

from .core import TritonCandidateProposal


class ReductionCandidateProvider:
    op_names = frozenset({"nn.softmax", "math.reduce_sum", "tensors.top_k", "nn.l2_normalization"})

    def propose(self, node, context):
        family = node.op.split(".", 1)[1]
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local"))
        if not candidates:
            return None
        return TritonCandidateProposal(
            candidates, context.choose_default(family, candidates, portable_fallback=f"tir.{family}.local"))
