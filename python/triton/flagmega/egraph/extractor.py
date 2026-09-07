# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""OR-Tools CP-SAT extraction for FlagMega e-graphs."""

from __future__ import annotations

import math
import os
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Callable, IO, Mapping

from triton.flagmega.diagnostics import DumpFlags, DumpScope
from triton.flagmega.egraph.graph import EClassView, EGraph, ENode, attrs_key_for
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, Node, verify_module


NodeCost = Callable[[Node, IRModule], float]


@dataclass(frozen=True)
class ForcedChoice:
    """A user/agent-selected representative that becomes a SAT constraint."""

    source_node: str
    node: Node


@dataclass(frozen=True)
class ExtractionResult:
    module: IRModule
    status: str
    objective: float
    costs: Mapping[ENode, float]
    picks: Mapping[ENode, bool]
    cost_model: str


class EGraphExtractor:
    """Globally extract a minimum-cost acyclic expression DAG with CP-SAT.

    The constraints follow nncase's extractor: a root must be selected and a
    selected e-node requires one representative from each child e-class. Rank
    constraints rule out cyclic representatives. At-most-one constraints make
    reconstruction well-defined even when a target assigns zero cost.
    """

    _COST_SCALE = 1_000_000

    def __init__(
        self,
        graph: EGraph,
        module: IRModule,
        *,
        cost: NodeCost | None = None,
        cost_model: str | None = None,
        forced: tuple[ForcedChoice, ...] = (),
    ) -> None:
        self.graph = graph
        self.module = module
        self.cost = cost or (lambda _node, _module: 1.0)
        self.cost_model = cost_model or (
            "structural-unit/v1" if cost is None else "custom-callable/unversioned"
        )
        if not self.cost_model:
            raise ValueError("EGraph extraction requires a cost model identifier.")
        self.forced = forced

    def extract(self) -> ExtractionResult:
        cp_model = _load_cp_model()
        roots = self.graph.roots_for_module(self.module)
        if not roots:
            return ExtractionResult(self.module, "EMPTY", 0.0, {}, {}, self.cost_model)
        classes = self.graph.reachable(roots)
        nodes = tuple(node for view in classes for node in view.nodes)
        costs = {node: self._node_cost(node) for node in nodes}

        model = cp_model.CpModel()
        variables = {
            node: model.NewBoolVar(f"e{view.id}_n{index}")
            for view in classes
            for index, node in enumerate(view.nodes)
        }
        class_map = {view.id: view for view in classes}

        # 1. Every externally observed root needs a representative.
        for root in roots:
            view = class_map[self.graph.find(root)]
            model.AddBoolOr([variables[node] for node in view.nodes])

        # 2. A selected node selects exactly one representative for each child.
        for view in classes:
            model.AddAtMostOne([variables[node] for node in view.nodes])
            for node in view.nodes:
                for child in node.children:
                    child_view = class_map[self.graph.find(child)]
                    model.AddBoolOr(
                        [variables[node].Not(), *[variables[value] for value in child_view.nodes]])

        # 3. Selected dependency edges must be strictly descending, which is a
        # compact CP-SAT encoding of nncase's no-cycle extraction constraint.
        rank = {
            view.id: model.NewIntVar(0, max(len(classes) - 1, 0), f"rank_e{view.id}")
            for view in classes
        }
        for view in classes:
            for node in view.nodes:
                for child in node.children:
                    child_id = self.graph.find(child)
                    model.Add(rank[view.id] > rank[child_id]).OnlyEnforceIf(variables[node])

        # Explicit agent/compiler selections remain constraints; the remaining
        # graph is still extracted globally by OR-Tools.
        for choice in self.forced:
            selected = self._resolve_choice(choice, class_map)
            model.Add(variables[selected] == 1)

        # CP-SAT uses integer coefficients. The large tie base preserves the
        # primary fixed-point cost, then deterministically prefers original and
        # earlier representatives when the target cost is equal.
        tie_bound = sum(sum(node.ordinal + 1 for node in view.nodes if not node.original) for view in classes)
        tie_base = tie_bound + 1
        coefficients = {
            node: int(round(costs[node] * self._COST_SCALE)) * tie_base
            + (0 if node.original else node.ordinal + 1)
            for node in nodes
        }
        model.Minimize(sum(coefficients[node] * variables[node] for node in nodes))
        validation = model.Validate()
        if validation:
            raise IRVerificationError(f"OR-Tools EGraph extraction model is invalid: {validation}")

        dumper = DumpScope.current()
        enable_cost_dump = dumper.is_enabled(DumpFlags.EGRAPH_COST) and dumper.directory is not None
        if enable_cost_dump:
            with dumper.open_artifact(
                "Costs/Cost.dot", category=DumpFlags.EGRAPH_COST,
                kind="egraph-dot", producer="EGraphExtractor",
                source_semantic_hash=self.module.semantic_hash, encoding="utf-8",
            ) as stream:
                stream.write(self.graph.to_dot(roots=roots, costs=costs))
            with dumper.open_artifact(
                "Costs/Cost.txt", category=DumpFlags.EGRAPH_COST,
                kind="cost-report", producer="EGraphExtractor",
                source_semantic_hash=self.module.semantic_hash, encoding="utf-8",
            ) as stream:
                stream.write(f"model: {self.cost_model}\n")
                _write_costs(stream, classes, costs)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = _positive_env_float("SOLVE_MAX_TIME", 120.0)
        default_workers = max((os.cpu_count() or 1) // 2, 1)
        solver.parameters.num_search_workers = _positive_env_int("SOLVE_PROCESSOR_COUNT", default_workers)
        with ExitStack() as artifacts:
            solve_stream = artifacts.enter_context(dumper.open_artifact(
                "Costs/Solve.txt", category=DumpFlags.EGRAPH_COST,
                kind="solver-log", producer="EGraphExtractor",
                source_semantic_hash=self.module.semantic_hash, encoding="utf-8",
            )) if enable_cost_dump else None
            callback = _CostCallback(cp_model, variables, costs, solve_stream)
            status_code = solver.Solve(model, callback)
            status = solver.StatusName(status_code)
            if solve_stream is not None:
                solve_stream.write(f"Status : {status}\n")
                solve_stream.write(f"Objective : {solver.ObjectiveValue()}\n")
                solve_stream.write(f"BestBound : {solver.BestObjectiveBound()}\n")

        if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise IRVerificationError(f"OR-Tools EGraph extraction failed with status {status}.")
        picks = {node: bool(solver.BooleanValue(variables[node])) for node in nodes}
        objective = sum(costs[node] for node, picked in picks.items() if picked)
        if enable_cost_dump:
            with dumper.open_artifact(
                "Costs/Pick.dot", category=DumpFlags.EGRAPH_COST,
                kind="egraph-pick-dot", producer="EGraphExtractor",
                source_semantic_hash=self.module.semantic_hash, encoding="utf-8",
            ) as stream:
                stream.write(self.graph.to_dot(roots=roots, costs=costs, picks=picks))
            with dumper.open_artifact(
                "Costs/Pick.txt", category=DumpFlags.EGRAPH_COST,
                kind="selection-report", producer="EGraphExtractor",
                source_semantic_hash=self.module.semantic_hash, encoding="utf-8",
            ) as stream:
                stream.write(
                    f"status: {status}\nmodel: {self.cost_model}\n"
                    f"objective: {objective:.12g}\n"
                )
                for node in sorted((node for node, picked in picks.items() if picked), key=lambda value: value.ordinal):
                    stream.write(f"pick n{node.ordinal}: {node.op} source={node.source_node} cost={costs[node]:.12g}\n")
        post = _materialize(self.graph, self.module, picks)
        return ExtractionResult(post, status, objective, costs, picks, self.cost_model)

    def _node_cost(self, node: ENode) -> float:
        value = 0.0 if node.op == "egraph.opaque" else float(self.cost(node.node, self.module))
        if not math.isfinite(value) or value < 0:
            raise IRVerificationError(
                f"EGraph cost for {node.op!r} must be finite and non-negative, got {value}.",
                node_id=node.source_node,
            )
        return value

    def _resolve_choice(self, choice: ForcedChoice, class_map: Mapping[int, EClassView]) -> ENode:
        class_id = self.graph.class_for_node(choice.source_node)
        view = class_map.get(class_id)
        if view is None:
            raise IRVerificationError(
                f"Forced EGraph choice for {choice.source_node!r} is unreachable.", node_id=choice.source_node)
        child_classes = tuple(self.graph.class_for_node(value) for value in choice.node.inputs)
        matches = tuple(
            node for node in view.nodes
            if node.op == choice.node.op
            and node.attrs_key == _attrs_key(choice.node)
            and tuple(self.graph.find(value) for value in node.children) == child_classes
        )
        if not matches:
            raise IRVerificationError(
                "E-graph selector returned a node that is not one of the alternatives.",
                stage=self.module.stage,
                node_id=choice.source_node,
            )
        return min(matches, key=lambda value: value.ordinal)


class _CostCallback:
    """Adapter because OR-Tools requires callbacks to inherit its runtime type."""

    def __new__(cls, cp_model, variables, costs, stream):
        class Callback(cp_model.CpSolverSolutionCallback):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            def on_solution_callback(self) -> None:
                if stream is None:
                    return
                objective = sum(costs[node] for node, variable in variables.items() if self.BooleanValue(variable))
                stream.write(f"Solution {self.count} @ {self.WallTime():.6f}:\n")
                stream.write(f"Cost: {objective:.12g}\n")
                stream.flush()
                self.count += 1

        return Callback()


def _materialize(graph: EGraph, module: IRModule, picks: Mapping[ENode, bool]) -> IRModule:
    selected: dict[int, ENode] = {}
    for view in graph.classes():
        chosen = tuple(node for node in view.nodes if picks.get(node, False))
        if len(chosen) > 1:
            raise IRVerificationError(f"E-class {view.id} selected more than one representative.")
        if chosen:
            selected[view.id] = chosen[0]

    original_order = {node.id: index for index, node in enumerate(module.nodes)}
    original_by_class: dict[int, list[str]] = {}
    for node in module.nodes:
        if graph.has_node_id(node.id):
            original_by_class.setdefault(graph.class_for_node(node.id), []).append(node.id)

    emitted: list[Node] = []
    emitted_ids: set[str] = set()
    class_values: dict[int, str] = {}
    original_values: dict[str, str] = {}
    active_classes: set[int] = set()
    active_originals: set[str] = set()

    def emit_class(class_id: int) -> str:
        class_id = graph.find(class_id)
        if class_id in class_values:
            return class_values[class_id]
        if class_id in active_classes:
            raise IRVerificationError(f"Selected EGraph expression contains a cycle through e-class {class_id}.")
        try:
            enode = selected[class_id]
        except KeyError as error:
            raise IRVerificationError(f"Required e-class {class_id} has no selected representative.") from error
        if enode.op == "egraph.opaque":
            return emit_original(enode.source_node)
        active_classes.add(class_id)
        inputs = tuple(emit_class(child) for child in enode.children)
        candidates = sorted(original_by_class.get(class_id, ()), key=original_order.__getitem__)
        node_id = next((value for value in candidates if value not in emitted_ids), enode.node.id)
        if node_id in emitted_ids:
            raise IRVerificationError(
                f"E-graph extraction generated colliding helper id {node_id!r}.", node_id=node_id)
        emitted.append(replace(enode.node, id=node_id, inputs=inputs))
        emitted_ids.add(node_id)
        class_values[class_id] = node_id
        for value in candidates:
            original_values[value] = node_id
        active_classes.remove(class_id)
        return node_id

    def emit_original(node_id: str) -> str:
        if node_id in original_values:
            return original_values[node_id]
        if node_id in active_originals:
            raise IRVerificationError(f"IR contains a cycle through node {node_id!r}.", node_id=node_id)
        node = module.node_map[node_id]
        if node.effect.is_pure and graph.has_node_id(node_id):
            value = emit_class(graph.class_for_node(node_id))
            original_values[node_id] = value
            return value
        active_originals.add(node_id)
        inputs = tuple(emit_original(value) for value in node.inputs)
        if node_id in emitted_ids:
            raise IRVerificationError(f"E-graph extraction generated duplicate node {node_id!r}.", node_id=node_id)
        emitted.append(replace(node, inputs=inputs))
        emitted_ids.add(node_id)
        original_values[node_id] = node_id
        active_originals.remove(node_id)
        return node_id

    for function in module.functions:
        for parameter in function.parameters:
            emit_original(parameter)
    for node in module.nodes:
        if not node.effect.is_pure:
            emit_original(node.id)
    functions = tuple(replace(
        function,
        parameters=tuple(emit_original(value) for value in function.parameters),
        outputs=tuple(emit_original(value) for value in function.outputs),
    ) for function in module.functions)
    owners = {
        node_id: original_values[node_id]
        for node_id in original_values
        if original_values[node_id] in emitted_ids
    }
    points = tuple(
        replace(point, owner=owners[point.owner]) if point.owner in owners else point
        for point in module.selection_points
        if point.owner is None or point.owner in owners
    )
    point_ids = {point.id for point in points}
    post = replace(
        module,
        nodes=tuple(emitted),
        functions=functions,
        selection_points=points,
        selections=tuple(record for record in module.selections if record.point_id in point_ids),
    )
    return verify_module(post)


def _attrs_key(node: Node) -> str:
    return attrs_key_for(node)


def _write_costs(stream: IO[str], classes: tuple[EClassView, ...], costs: Mapping[ENode, float]) -> None:
    for view in classes:
        stream.write(f"eclass {view.id}:\n")
        for node in sorted(view.nodes, key=lambda value: value.ordinal):
            stream.write(
                f"  n{node.ordinal}: op={node.op} source={node.source_node} "
                f"cost={costs[node]:.12g} original={str(node.original).lower()}\n")


def _load_cp_model():
    try:
        from ortools.sat.python import cp_model
    except ImportError as error:
        raise RuntimeError(
            "FlagMega EGraph extraction requires OR-Tools CP-SAT; install the FlagTree package dependencies "
            "or run `python -m pip install ortools==9.10.4067`."
        ) from error
    return cp_model


def _positive_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


__all__ = ["EGraphExtractor", "ExtractionResult", "ForcedChoice", "NodeCost"]
