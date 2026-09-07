# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Registry-driven Triton TIR candidate selection."""

from __future__ import annotations

from collections.abc import Callable

from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateContext,
    TritonCandidateProposal,
    TritonCandidateProviderRegistry,
    default_triton_candidate_registry,
)
from triton.flagmega.codegen.triton.distribution import requires_distributed_grid
from triton.flagmega.ir import Candidate, IRModule, SelectionPoint
from triton.flagmega.passes.tir import (
    find_norm_consumer_matches,
    find_projection_logits_argmax_matches,
    find_projection_residual_norm_matches,
)
from triton.flagmega.passes.functions import reusable_function_node_ids

WorkspaceAnnotator = Callable[..., tuple[Candidate, ...]]


class TritonTirSelectionPolicy:
    """Orchestrate independent candidate-family providers.

    Providers own semantic-op recognition and candidate construction. This
    policy owns only graph analyses, workspace annotation, selection-point
    construction, and target capability filtering.
    """

    def __init__(
        self,
        workspace_annotator: WorkspaceAnnotator,
        registry: TritonCandidateProviderRegistry | None = None,
    ) -> None:
        self.workspace_annotator = workspace_annotator
        self.registry = registry or default_triton_candidate_registry()

    @property
    def op_names(self) -> frozenset[str]:
        return self.registry.op_names

    def propose(self, module: IRModule, target) -> IRModule:
        existing = {point.id for point in module.selection_points}
        context = TritonCandidateContext(
            module=module,
            target=target,
            projection_norm_matches=find_projection_residual_norm_matches(module),
            projection_logits_matches=find_projection_logits_argmax_matches(module),
            norm_consumer_matches=find_norm_consumer_matches(module),
            reusable_node_ids=reusable_function_node_ids(module),
            cooperative_grid=requires_distributed_grid(module),
            implementation_model=target.triton_implementation_model,
        )
        mesh_hierarchy = target.distributed_placements(module)[0].hierarchy
        points: list[SelectionPoint] = []
        for node in module.nodes:
            point_id = f"tir.{node.id}"
            if point_id in existing:
                continue
            provider = self.registry.provider_for(node.op)
            if provider is None:
                continue
            proposal = provider.propose(node, context)
            if proposal is None:
                continue
            candidates = self.workspace_annotator(
                node,
                proposal.candidates,
                module,
                mesh_hierarchy=mesh_hierarchy,
            )
            # Validate callbacks at the policy boundary. A target may annotate
            # facts, but may not silently remove the provider's default.
            proposal = TritonCandidateProposal(
                candidates,
                proposal.default_candidate,
                proposal.selection_kind,
            )
            points.append(SelectionPoint(
                point_id,
                proposal.selection_kind,
                proposal.candidates,
                proposal.default_candidate,
                owner=node.id,
            ))
        return target.add_default_selections(
            module,
            tuple(points),
            (
                "Prefer resource-feasible Triton kernel variants supported "
                "by the active target."
            ),
            policy_version=target.policy_version,
        )


__all__ = ["TritonTirSelectionPolicy", "WorkspaceAnnotator"]
