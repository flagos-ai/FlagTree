# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Equality saturation sessions shared by nncase-shaped EGraph passes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

from triton.flagmega.diagnostics import DumpFlags, DumpScope
from triton.flagmega.egraph.extractor import EGraphExtractor, ExtractionResult, ForcedChoice, NodeCost
from triton.flagmega.egraph.graph import EGraph, ENode
from triton.flagmega.egraph.matcher import find_egraph_matches
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, Node, verify_module
from triton.flagmega.rules import RewriteRedirect, RewriteResult, RewriteRule


@dataclass(frozen=True)
class EGraphAlternative:
    rule: str
    result: RewriteResult
    cost: float

    @property
    def node(self) -> Node:
        """Compatibility view used by existing selector callbacks."""

        return self.result.replacement


@dataclass(frozen=True)
class RewriteIteration:
    version_before: int
    version_after: int
    matches: tuple[tuple[str, int], ...]


AlternativeSelector = Callable[[Node, tuple[EGraphAlternative, ...], IRModule], Node]


class EGraphSession:
    """The graph state spanning Construct -> rule passes -> Extract."""

    def __init__(self, *, node_limit: int = 10000, class_limit: int = 5000) -> None:
        self.node_limit = node_limit
        self.class_limit = class_limit
        self.module: IRModule | None = None
        self.graph: EGraph | None = None
        self.cost: NodeCost | None = None
        self.cost_model: str | None = None
        self.forced: list[ForcedChoice] = []
        self.last_extraction: ExtractionResult | None = None

    def construct(self, module: IRModule) -> None:
        from triton.flagmega.passes.constants import require_constants_open

        require_constants_open(module, "EGraphConstructPass")
        self.module = verify_module(module)
        self.graph = EGraph(node_limit=self.node_limit, class_limit=self.class_limit)
        self.graph.add_module(self.module)
        self.cost = None
        self.cost_model = None
        self.forced.clear()
        self.last_extraction = None

    def apply_rules(
        self,
        name: str,
        rules: tuple[RewriteRule, ...],
        *,
        cost: NodeCost | None = None,
        cost_model: str | None = None,
        selector: AlternativeSelector | None = None,
        max_iterations: int = 32,
    ) -> tuple[RewriteIteration, ...]:
        module, graph = self._require_constructed()
        if max_iterations <= 0:
            raise ValueError("EGraphRulesPass max_iterations must be positive.")
        if cost_model is not None and cost is None:
            raise ValueError("An EGraph cost_model name requires a cost callable.")
        if cost is not None:
            if self.cost is not None and self.cost is not cost:
                raise IRVerificationError(
                    "Contiguous EGraphRulesPass instances must share one extraction cost model.",
                    stage=module.stage,
                )
            self.cost = cost
            resolved_model = cost_model or "custom-callable/unversioned"
            if self.cost_model is not None and self.cost_model != resolved_model:
                raise IRVerificationError(
                    "Contiguous EGraphRulesPass instances must share one cost model identifier.",
                    stage=module.stage,
                )
            self.cost_model = resolved_model
        if selector is not None:
            return (self._apply_selected_once(name, rules, cost, selector),)

        iterations: list[RewriteIteration] = []
        for _ in range(max_iterations):
            before = graph.version
            counts = [0] * len(rules)
            module_view = self._module_view()
            pending: list[tuple[RewriteRule, Node, object]] = []
            # Match every rule against one immutable e-graph snapshot.  A
            # subsequent iteration is required to observe alternatives added
            # by this batch, exactly as in nncase's EGraphRewriteProvider.
            enodes = tuple(
                enode for _class_id, enode in graph.enodes()
                if enode.op != "egraph.opaque"
            )
            for rule_index, rule in enumerate(rules):
                if rule.pattern is not None:
                    for match in find_egraph_matches(graph, rule.pattern, module_view):
                        rewritten = rule.apply_match(match, module_view)
                        if rewritten is None:
                            continue
                        counts[rule_index] += 1
                        pending.append((rule, match.root, rewritten))
                    continue
                for enode in enodes:
                    rewritten = rule.apply(enode.node, module_view)
                    if rewritten is None:
                        continue
                    counts[rule_index] += 1
                    pending.append((rule, enode.node, rewritten))
            for rule, source, rewritten in pending:
                self._add_output(rule.name, source, rewritten)
            graph.rebuild()
            iteration = RewriteIteration(
                before,
                graph.version,
                tuple((rule.name, count) for rule, count in zip(rules, counts)),
            )
            iterations.append(iteration)
            self._dump_iteration(iteration)
            if graph.version == before:
                return tuple(iterations)
        raise IRVerificationError(
            f"EGraph rewrite pass {name!r} did not saturate after {max_iterations} iterations.",
            stage=module.stage,
        )

    def extract(self) -> IRModule:
        module, graph = self._require_constructed()
        self.last_extraction = EGraphExtractor(
            graph,
            module,
            cost=self.cost,
            cost_model=self.cost_model,
            forced=tuple(self.forced),
        ).extract()
        return self.last_extraction.module

    def _apply_selected_once(
        self,
        _name: str,
        rules: tuple[RewriteRule, ...],
        cost: NodeCost | None,
        selector: AlternativeSelector,
    ) -> RewriteIteration:
        module, graph = self._require_constructed()
        before = graph.version
        counts = [0] * len(rules)
        for node in module.nodes:
            if not node.effect.is_pure:
                continue
            alternatives: list[EGraphAlternative] = []
            for rule_index, rule in enumerate(rules):
                rewritten = rule.apply(node, module)
                if rewritten is None:
                    continue
                counts[rule_index] += 1
                result = rewritten if isinstance(rewritten, RewriteResult) else RewriteResult(rewritten)
                self._add_result(rule.name, node, result)
                score = float((cost or self.cost or (lambda _node, _module: 1.0))(result.replacement, module))
                alternatives.append(EGraphAlternative(rule.name, result, score))
            if not alternatives:
                continue
            selected = selector(node, tuple(alternatives), module)
            if selected.id != node.id:
                selected = replace(selected, id=node.id)
            known = (node, *(value.node for value in alternatives))
            if selected not in known:
                raise IRVerificationError(
                    "E-graph selector returned a node that is not one of the alternatives.",
                    stage=module.stage,
                    node_id=node.id,
                )
            self.forced.append(ForcedChoice(node.id, selected))
        graph.rebuild()
        iteration = RewriteIteration(
            before,
            graph.version,
            tuple((rule.name, count) for rule, count in zip(rules, counts)),
        )
        self._dump_iteration(iteration)
        return iteration

    def _add_output(self, rule_name: str, source: Node, output: object) -> None:
        module, graph = self._require_constructed()
        if isinstance(output, RewriteRedirect):
            if not graph.has_node_id(output.target_id):
                raise IRVerificationError(
                    f"E-graph rule {rule_name!r} redirects to unknown node {output.target_id!r}.",
                    stage=module.stage,
                    node_id=source.id,
                )
            graph.union(
                graph.class_for_node(source.id),
                graph.class_for_node(output.target_id),
            )
            return
        result = output if isinstance(output, RewriteResult) else RewriteResult(output)
        self._add_result(rule_name, source, result)

    def _add_result(self, rule_name: str, source: Node, result: RewriteResult) -> None:
        module, graph = self._require_constructed()
        if result.replacement.type != source.type or not result.replacement.effect.is_pure:
            raise IRVerificationError(
                f"E-graph rule {rule_name!r} must preserve type and purity.",
                stage=module.stage,
                node_id=source.id,
            )
        if result.replacement.id != source.id:
            raise IRVerificationError(
                f"E-graph rule {rule_name!r} must preserve root id {source.id!r}.",
                stage=module.stage,
                node_id=source.id,
            )
        _verify_helpers(result, source, module, graph)
        for helper in result.prefix_nodes:
            graph.add_node(helper)
        graph.add_equivalent(source.id, result.replacement)

    def _module_view(self) -> IRModule:
        module, graph = self._require_constructed()
        original_ids = set(module.node_map)
        helpers: dict[str, Node] = {}
        for _class_id, enode in graph.enodes():
            if enode.node.id not in original_ids:
                helpers.setdefault(enode.node.id, enode.node)
        return replace(module, nodes=(*module.nodes, *helpers.values()))

    def _dump_iteration(self, iteration: RewriteIteration) -> None:
        _module, graph = self._require_constructed()
        dumper = DumpScope.current()
        if dumper.is_enabled(DumpFlags.REWRITE) and dumper.directory is not None:
            with dumper.open_artifact(
                f"Matches/V{iteration.version_before}.txt",
                category=DumpFlags.REWRITE,
                kind="rewrite-matches",
                producer="EGraphSession",
                source_semantic_hash=self.module.semantic_hash,
                encoding="utf-8",
            ) as stream:
                stream.write("rule, results\n")
                for rule, count in iteration.matches:
                    stream.write(f"{rule}, {count}\n")
            if iteration.version_after != iteration.version_before:
                with dumper.open_artifact(
                    f"Rebuild/V{iteration.version_after}.dot",
                    category=DumpFlags.REWRITE,
                    kind="egraph-rebuild-dot",
                    producer="EGraphSession",
                    source_semantic_hash=self.module.semantic_hash,
                    encoding="utf-8",
                ) as stream:
                    stream.write(graph.to_dot(roots=graph.roots_for_module(self.module)))

    def _require_constructed(self) -> tuple[IRModule, EGraph]:
        if self.module is None or self.graph is None:
            raise RuntimeError("EGraph session has not been constructed.")
        return self.module, self.graph


class EGraphRewriter:
    """Standalone compatibility facade over the three-pass lifecycle."""

    def __init__(
        self,
        rules: Iterable[RewriteRule],
        *,
        cost: NodeCost | None = None,
        cost_model: str | None = None,
        selector: AlternativeSelector | None = None,
        node_limit: int = 10000,
        class_limit: int = 5000,
        max_iterations: int = 32,
    ) -> None:
        self.rules = tuple(rules)
        self.cost = cost
        self.cost_model = cost_model
        self.selector = selector
        self.node_limit = node_limit
        self.class_limit = class_limit
        self.max_iterations = max_iterations
        self.last_graph: EGraph | None = None
        self.last_extraction: ExtractionResult | None = None

    def rewrite(self, module: IRModule) -> IRModule:
        session = EGraphSession(node_limit=self.node_limit, class_limit=self.class_limit)
        session.construct(module)
        session.apply_rules(
            "EGraphRulesPass",
            self.rules,
            cost=self.cost,
            cost_model=self.cost_model,
            selector=self.selector,
            max_iterations=self.max_iterations,
        )
        result = session.extract()
        self.last_graph = session.graph
        self.last_extraction = session.last_extraction
        return result


def _verify_helpers(result: RewriteResult, root: Node, module: IRModule, graph: EGraph) -> None:
    occupied = set(module.node_map)
    generated: set[str] = set()
    for helper in result.prefix_nodes:
        if not helper.effect.is_pure:
            raise IRVerificationError(
                "E-graph alternatives cannot insert effectful helper nodes.",
                stage=module.stage,
                node_id=root.id,
            )
        if helper.id in occupied or helper.id in generated:
            raise IRVerificationError(
                f"E-graph extraction generated colliding helper ids {[helper.id]!r}.",
                stage=module.stage,
                node_id=root.id,
            )
        missing = {value for value in helper.inputs if not graph.has_node_id(value) and value not in generated}
        if missing:
            raise IRVerificationError(
                f"E-graph helper {helper.id!r} has non-topological inputs {sorted(missing)}.",
                stage=module.stage,
                node_id=root.id,
            )
        generated.add(helper.id)
    missing = {
        value for value in result.replacement.inputs
        if not graph.has_node_id(value) and value not in generated
    }
    if missing:
        raise IRVerificationError(
            f"E-graph replacement {root.id!r} has non-topological inputs {sorted(missing)}.",
            stage=module.stage,
            node_id=root.id,
        )


__all__ = [
    "AlternativeSelector",
    "EGraphAlternative",
    "EGraphRewriter",
    "EGraphSession",
    "NodeCost",
    "RewriteIteration",
]
