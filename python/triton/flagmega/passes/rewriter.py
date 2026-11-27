# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""nncase-shaped rewrite passes accepted directly by PassManager."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from triton.flagmega.diagnostics import DumpFlags, DumpScope
from triton.flagmega.egraph import AlternativeSelector, EGraphRewriter, EGraphSession, NodeCost
from triton.flagmega.ir import IRModule
from triton.flagmega.rules import DataflowRewriter, RewriteRule


@dataclass(frozen=True)
class DataflowPass:
    name: str
    rules: tuple[RewriteRule, ...]
    max_iterations: int = 32
    remove_unused: bool = True
    preserves: frozenset[str] = frozenset()
    rewrite_constants: bool = True

    def run(self, module: IRModule) -> IRModule:
        return DataflowRewriter(
            self.rules,
            max_iterations=self.max_iterations,
            remove_unused=self.remove_unused,
            rewrite_constants=self.rewrite_constants,
        ).rewrite(module)


@dataclass(frozen=True)
class EGraphRulesPass:
    name: str
    rules: tuple[RewriteRule, ...]
    cost: NodeCost | None = None
    cost_model: str | None = None
    selector: AlternativeSelector | None = None
    node_limit: int = 10000
    class_limit: int = 5000
    max_iterations: int = 32
    preserves: frozenset[str] = frozenset()
    _session: EGraphSession | None = field(default=None, compare=False, repr=False)

    def run(self, module: IRModule) -> IRModule:
        if self._session is not None:
            graph = self._session.graph
            if graph is None:
                raise RuntimeError("EGraphRulesPass ran before EGraphConstructPass.")
            _dump_graph("Start", graph, module)
            self._session.apply_rules(
                self.name,
                self.rules,
                cost=self.cost,
                cost_model=self.cost_model,
                selector=self.selector,
                max_iterations=self.max_iterations,
            )
            _dump_graph("End", graph, module)
            return module
        return EGraphRewriter(
            self.rules,
            cost=self.cost,
            cost_model=self.cost_model,
            selector=self.selector,
            node_limit=self.node_limit,
            class_limit=self.class_limit,
            max_iterations=self.max_iterations,
        ).rewrite(module)

    def bind(self, session: EGraphSession) -> EGraphRulesPass:
        return replace(self, _session=session)


@dataclass(frozen=True)
class EGraphConstructPass:
    session: EGraphSession = field(repr=False, compare=False)
    name: str = "EGraphConstructPass"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        self.session.construct(module)
        assert self.session.graph is not None
        _dump_graph("End", self.session.graph, module)
        return module


@dataclass(frozen=True)
class EGraphExtractPass:
    session: EGraphSession = field(repr=False, compare=False)
    name: str = "EGraphExtractPass"
    preserves: frozenset[str] = frozenset()

    def run(self, module: IRModule) -> IRModule:
        if self.session.graph is None:
            raise RuntimeError("EGraphExtractPass ran without EGraphConstructPass.")
        _dump_graph("Start", self.session.graph, module)
        return self.session.extract()


def _dump_graph(prefix: str, graph, module: IRModule) -> None:
    dumper = DumpScope.current()
    if dumper.is_enabled(DumpFlags.PASS_IR) and dumper.directory is not None:
        with dumper.open_artifact(
            f"{prefix}/V{graph.version}.dot",
            category=DumpFlags.PASS_IR,
            kind="egraph-boundary-dot",
            producer="EGraphPass",
            source_semantic_hash=module.semantic_hash,
            encoding="utf-8",
        ) as stream:
            stream.write(graph.to_dot(roots=graph.roots_for_module(module)))


__all__ = ["DataflowPass", "EGraphConstructPass", "EGraphExtractPass", "EGraphRulesPass"]
