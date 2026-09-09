# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from .core import TritonCandidateProposal


class DeltaRuleCandidateProvider:
    op_names = frozenset({"nn.delta_rule_coefficients", "nn.delta_rule_log_prefix", "nn.delta_rule_block_update",
                          "nn.delta_rule_gates"})

    def propose(self, node, context):
        family = node.op.removeprefix("nn.")
        candidates = tuple(
            context.configure_implementation(implementation)
            for implementation in context.implementations(family, indexing="local"))
        if not candidates:
            return None
        return TritonCandidateProposal(candidates, context.choose_default(family, candidates))
