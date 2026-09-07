# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-independent capability filtering and selection provenance."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, SelectionPoint, SelectionRecord


class CandidateCapability(Protocol):
    def supports(self, requirements) -> bool: ...


class CapabilitySelectionPolicy:
    policy_identity: str | None = None

    def choose_default(
        self,
        module: IRModule,
        point: SelectionPoint,
        candidates,
    ) -> str:
        """Choose among already capability-filtered candidates.

        Semantic candidate providers own the portable default.  Physical
        machines may override this hook without moving their preference into
        the rule that constructed the candidates.
        """

        candidate_ids = {candidate.id for candidate in candidates}
        return (
            point.default_candidate
            if point.default_candidate in candidate_ids
            else candidates[0].id
        )

    def add_defaults(
        self,
        module: IRModule,
        points: tuple[SelectionPoint, ...],
        rationale: str,
        *,
        capability: CandidateCapability,
        target_name: str,
        policy_version: str,
    ) -> IRModule:
        if not points:
            return module
        supported_points = []
        selected_policies: dict[str, str] = {}
        for point in points:
            candidates = tuple(
                candidate
                for candidate in point.candidates
                if capability.supports(candidate_requirements(candidate))
            )
            if not candidates:
                requirements = {
                    candidate.id: candidate_requirements(candidate)
                    for candidate in point.candidates
                }
                raise IRVerificationError(
                    f"Target {target_name!r} selection point {point.id!r} has no "
                    "capability-supported "
                    f"candidate; requirements={requirements}.",
                    stage=module.stage,
                )
            semantic_default = point.default_candidate
            default = self.choose_default(module, point, candidates)
            if default not in {candidate.id for candidate in candidates}:
                raise IRVerificationError(
                    f"Target selection policy returned unavailable candidate {default!r} "
                    f"for {point.id!r}.",
                    stage=module.stage,
                )
            supported_points.append(replace(
                point,
                candidates=candidates,
                default_candidate=default,
            ))
            selection_policy = (
                self.policy_identity
                if default != semantic_default and self.policy_identity is not None
                else policy_version
            )
            selected_policies[point.id] = selection_policy
        filtered = tuple(supported_points)
        selections = module.selections + tuple(
            SelectionRecord(
                point_id=point.id,
                candidate_id=point.default_candidate,
                origin="default-policy",
                policy=selected_policies[point.id],
                rationale=rationale,
            )
            for point in filtered
        )
        return replace(
            module,
            selection_points=module.selection_points + filtered,
            selections=selections,
        )


def candidate_requirements(candidate) -> tuple[str, ...]:
    value = candidate.facts.get("requires", ())
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


__all__ = ["CandidateCapability", "CapabilitySelectionPolicy", "candidate_requirements"]
