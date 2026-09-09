# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Proposal graph lifetime is owned by the ordinary pass analysis manager."""

from dataclasses import dataclass, fields, replace

from triton.flagmega.ir import IRModule
from triton.flagmega.passes.context import current_pass_context

PROPOSAL_ANALYSIS = "auto-distributed-proposal-search"


@dataclass(frozen=True)
class ProposalAnalysis:
    module: IRModule
    graph: object
    target: object
    policy_identity: str

    def rebind(self, module, target):
        if target is not self.target or target.distribution_policy.identity != self.policy_identity:
            return None
        # Stage.run adds the named boundary and provenance after propose.
        # Everything editable, including selections/metadata/constant recipes,
        # must still equal the exact proposal for which this graph was built.
        if module.stage not in (self.module.stage, "distribution_candidates"):
            return None
        if any(
                getattr(module, item.name) != getattr(self.module, item.name)
                for item in fields(IRModule)
                if item.name not in {"stage", "provenance"}):
            return None
        graph = replace(self.graph, module=module)
        graph._realized_costs.update(self.graph._realized_costs)
        return graph


def remember_proposal(module, graph, target):
    try:
        context = current_pass_context()
    except RuntimeError:
        return
    context.analyses.seed(PROPOSAL_ANALYSIS, ProposalAnalysis(module, graph, target,
                                                              target.distribution_policy.identity))


def proposal_graph(module, target):
    try:
        context = current_pass_context()
    except RuntimeError:
        return None
    if PROPOSAL_ANALYSIS not in context.analyses.cached_names:
        return None
    analysis = context.require_analysis(PROPOSAL_ANALYSIS)
    return analysis.rebind(module, target) if isinstance(analysis, ProposalAnalysis) else None


__all__ = ["PROPOSAL_ANALYSIS"]
