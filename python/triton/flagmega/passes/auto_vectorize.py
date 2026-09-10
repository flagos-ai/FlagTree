# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped EGraph AutoVectorize orchestration."""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.diagnostics.dump import DumpScope
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import Candidate, IRModule, SelectionPoint, SelectionRecord
from triton.flagmega.passes.manager import PassManager
from triton.flagmega.passes.rewriter import DataflowPass, EGraphRulesPass
from triton.flagmega.passes.vector_layout import coordinate_layouts, region_candidates
from triton.flagmega.rules import RewriteRule
from triton.flagmega.rules.ntt.vectorize import VectorizeCandidate, VectorizeRuleRegistry


class AutoVectorizePass:
    """Generate editable candidates, then extract selected EGraph alternatives."""

    @staticmethod
    def propose(module: IRModule, target) -> IRModule:
        registry = _registry(target)
        existing = {point.id for point in module.selection_points}
        points: list[SelectionPoint] = []
        selections: list[SelectionRecord] = []
        alternatives_by_node, defaults = coordinate_layouts(module, registry, {
            node.id: tuple(candidate for rule in registry.rules for candidate in rule.candidates(node, module))
            for node in module.nodes
        }, _default_candidate)
        for node in module.nodes:
            alternatives = alternatives_by_node[node.id]
            if not alternatives:
                continue
            point_id = f"vectorization.{node.id}"
            if point_id in existing:
                continue
            candidates = (Candidate(
                "vectorization.scalar",
                {"strategy": "scalar"},
                {"egraph_original": True},
            ), *(_to_ir_candidate(value) for value in alternatives))
            default = defaults[node.id]
            points.append(SelectionPoint(point_id, "vectorization", candidates, default, owner=node.id))
            selections.append(SelectionRecord(
                point_id,
                default,
                "default-policy",
                target.vectorization_policy.identity,
                "Extract the target-preferred typed-vector EGraph alternative; the Python checkpoint may override it.",
            ))
        return replace(
            module,
            selection_points=module.selection_points + tuple(points),
            selections=module.selections + tuple(selections),
        )

    @staticmethod
    def run(module: IRModule, target) -> IRModule:
        registry = _registry(target)
        selected = module.selection_map
        pure_rules: list[RewriteRule] = []
        effectful_rules: list[RewriteRule] = []
        for point in module.selection_points:
            if point.kind != "vectorization" or point.owner is None:
                continue
            record = selected.get(point.id)
            if record is None:
                raise IRVerificationError(
                    f"Vectorization selection point {point.id!r} has no applied selection.",
                    stage=module.stage,
                    node_id=point.owner,
                )
            if record.candidate_id == "vectorization.scalar":
                continue
            ir_candidate = next(value for value in point.candidates if value.id == record.candidate_id)
            rule_name = str(ir_candidate.parameters["rule"])
            rule = next((value for value in registry.rules if value.name == rule_name), None)
            if rule is None:
                raise IRVerificationError(
                    f"Target did not register selected vectorization rule {rule_name!r}.",
                    stage=module.stage,
                    node_id=point.owner,
                )
            node = module.node_map[point.owner]
            candidates = (region_candidates(rule, node, module, ir_candidate.parameters["region_lane_bytes"],
                                            ir_candidate.parameters.get("region_lanes"))
                          if "region_lane_bytes" in ir_candidate.parameters else rule.candidates(node, module))
            candidate = next(
                (value for value in candidates if value.id == record.candidate_id),
                None,
            )
            if candidate is None:
                raise IRVerificationError(
                    f"Selected vectorization candidate {record.candidate_id!r} is no longer legal.",
                    stage=module.stage,
                    node_id=point.owner,
                )
            rewrite = RewriteRule(
                f"{rule.name}:{point.owner}:{candidate.id}",
                lambda value, _module, owner=point.owner: (
                    value.id == owner and "vectorized_from" not in value.metadata
                ),
                lambda value, current, selected_rule=rule, selected_candidate=candidate: (
                    selected_rule.rewrite(value, current, selected_candidate)
                ),
            )
            # Equality saturation deliberately represents an effectful node as
            # an opaque boundary: merging it into an e-class could duplicate,
            # delete, or reorder state updates.  A user-selected vectorization
            # is no longer an equality-search decision, however.  Materialize
            # that one local replacement with DataflowRewriter, whose default
            # PRESERVE policy verifies the root id/effect and rejects effectful
            # helpers.  Keeping this routing here also prevents a selection
            # record from claiming that an opaque rewrite was applied when it
            # was silently skipped by the e-graph.
            (pure_rules if node.effect.is_pure else effectful_rules).append(rewrite)

        current = module
        if pure_rules:
            # This is deliberately EGraphRulesPass, matching nncase.  Agent or
            # default selection has already chosen among equality candidates.
            egraph_dumper = DumpScope.current().create_sub_dumper("AutoVectorizeEGraph")
            current = PassManager("AutoVectorizeEGraph", dumper=egraph_dumper).add(EGraphRulesPass(
                "AutoVectorize",
                tuple(pure_rules),
                selector=lambda _original, alternatives, _module: alternatives[0].node,
            )).run(current).module
        if effectful_rules:
            current = DataflowPass(
                "ApplyEffectfulVectorization",
                tuple(effectful_rules),
                max_iterations=2,
            ).run(current)
        if registry.propagation_rules:
            current = DataflowPass(
                "PackPropagation",
                registry.propagation_rules,
                max_iterations=128,
            ).run(current)
        return current


def _registry(target) -> VectorizeRuleRegistry:
    registry = VectorizeRuleRegistry()
    target.register_auto_vectorize_rules(registry)
    target.register_pack_propagation_rules(registry)
    return registry


def _to_ir_candidate(value: VectorizeCandidate) -> Candidate:
    return Candidate(
        value.id,
        {**value.parameters, "rule": value.rule},
        value.facts,
    )


def _default_candidate(candidates: tuple[VectorizeCandidate, ...]) -> str:
    for preferred in ("vectorization.last_axis", "vectorization.matmul.n", "vectorization.matmul.k"):
        if any(value.id == preferred for value in candidates):
            return preferred
    return candidates[0].id


__all__ = ["AutoVectorizePass"]
