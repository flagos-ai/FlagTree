# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed congruence graph used by FlagMega equality saturation."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, IRType, Node


@dataclass(frozen=True)
class ENode:
    """An expression operator whose operands are e-classes."""

    op: str
    children: tuple[int, ...]
    attrs_key: str
    source_node: str
    node: Node = field(compare=False, hash=False, repr=False)
    ordinal: int = field(compare=False, hash=False, default=0)
    original: bool = field(compare=False, hash=False, default=False)

    @property
    def key(self) -> tuple[str, tuple[int, ...], str]:
        return self.op, self.children, self.attrs_key


@dataclass(frozen=True)
class EClassView:
    id: int
    type: IRType
    nodes: tuple[ENode, ...]


class EGraph:
    """Bounded, typed e-graph for effect-free FlagMega expressions.

    Effectful values are represented by opaque leaves. They remain outside
    equality saturation and are restored from the source module by extraction.
    """

    def __init__(self, *, node_limit: int = 10000, class_limit: int = 5000) -> None:
        if node_limit <= 0 or class_limit <= 0:
            raise ValueError("E-graph limits must be positive.")
        self.node_limit = node_limit
        self.class_limit = class_limit
        self.version = 0
        self._parent: list[int] = []
        self._types: list[IRType] = []
        self._nodes: list[list[ENode]] = []
        self._node_class: dict[str, int] = {}
        self._source_nodes: dict[str, Node] = {}
        self._memo: dict[tuple[str, tuple[int, ...], str], int] = {}
        self._ordinal = 0

    def add_module(self, module: IRModule) -> dict[str, int]:
        for node in module.nodes:
            if not node.effect.is_pure:
                continue
            for input_id in node.inputs:
                if input_id not in self._node_class:
                    self._add_leaf(module.node_map[input_id])
            self.add_node(node, original=True)
        return {node_id: self.find(class_id) for node_id, class_id in self._node_class.items()}

    def _add_leaf(self, node: Node) -> int:
        # Effectful expressions are boundaries, not ordinary constants.  Their
        # identity is part of the value: two writes produced by the same op
        # may observe different state and must never become congruent merely
        # because their result types and operator names match.
        leaf = Node(
            id=node.id,
            op="egraph.opaque",
            inputs=(),
            type=node.type,
            attrs={"source_op": node.op, "source_id": node.id},
        )
        return self.add_node(leaf, original=True)

    def add_node(self, node: Node, *, original: bool = False) -> int:
        existing_source = self._node_class.get(node.id)
        children = tuple(self.class_for_node(input_id) for input_id in node.inputs)
        class_id, _ = self._add_enode(node, children, original=original)
        if existing_source is not None and self.find(existing_source) != self.find(class_id):
            raise IRVerificationError(
                f"E-graph source id {node.id!r} denotes two non-equivalent expressions.",
                node_id=node.id,
            )
        self._node_class[node.id] = class_id
        # Hash-consing can discard this e-node while later helpers still name
        # its source id. Retain the original expression for dataflow-rule
        # snapshots, independently of canonical e-node storage.
        self._source_nodes.setdefault(node.id, node)
        return self.find(class_id)

    def add_equivalent(self, source_node_id: str, alternative: Node) -> tuple[int, ENode, bool]:
        """Add ``alternative`` and union it with ``source_node_id``."""

        source_class = self.class_for_node(source_node_id)
        children = tuple(self.class_for_node(input_id) for input_id in alternative.inputs)
        class_id, enode = self._add_enode(alternative, children, original=False)
        already_equivalent = self.find(source_class) == self.find(class_id)
        result = self.union(source_class, class_id)
        return result, enode, not already_equivalent

    def _add_enode(
        self,
        node: Node,
        children: tuple[int, ...],
        *,
        original: bool,
    ) -> tuple[int, ENode]:
        children = tuple(self.find(child) for child in children)
        attrs_key = attrs_key_for(node)
        key = (node.op, children, attrs_key)
        if key in self._memo:
            class_id = self.find(self._memo[key])
            if self._types[class_id] != node.type:
                raise IRVerificationError(f"Congruent e-nodes for {node.op!r} have different types.")
            existing = next(value for value in self._nodes[class_id] if value.key == key)
            return class_id, existing
        self._check_limits(node.id)
        class_id = len(self._parent)
        enode = ENode(
            node.op,
            children,
            attrs_key,
            node.id,
            node,
            self._ordinal,
            original,
        )
        self._ordinal += 1
        self._parent.append(class_id)
        self._types.append(node.type)
        self._nodes.append([enode])
        self._memo[key] = class_id
        self.version += 1
        return class_id, enode

    def _check_limits(self, node_id: str) -> None:
        if len(self._parent) >= self.class_limit:
            raise IRVerificationError(f"E-graph class limit {self.class_limit} exceeded.", node_id=node_id)
        if self.node_count >= self.node_limit:
            raise IRVerificationError(f"E-graph node limit {self.node_limit} exceeded.", node_id=node_id)

    @property
    def node_count(self) -> int:
        return sum(len(nodes) for class_id, nodes in enumerate(self._nodes) if self.find(class_id) == class_id)

    def find(self, class_id: int) -> int:
        parent = self._parent[class_id]
        if parent != class_id:
            self._parent[class_id] = self.find(parent)
        return self._parent[class_id]

    def union(self, lhs: int, rhs: int) -> int:
        lhs = self.find(lhs)
        rhs = self.find(rhs)
        if lhs == rhs:
            return lhs
        if self._types[lhs] != self._types[rhs]:
            raise IRVerificationError("Cannot union e-classes with different IR types.")
        root, merged = (lhs, rhs) if lhs < rhs else (rhs, lhs)
        self._parent[merged] = root
        self._nodes[root].extend(self._nodes[merged])
        self.version += 1
        self.rebuild()
        return self.find(root)

    def rebuild(self) -> None:
        """Restore hash-consing after unions and close congruence."""

        changed = False
        while True:
            new_memo: dict[tuple[str, tuple[int, ...], str], int] = {}
            pending_unions: list[tuple[int, int]] = []
            for class_id, nodes in enumerate(self._nodes):
                root = self.find(class_id)
                if root != class_id:
                    continue
                rewritten: list[ENode] = []
                for node in nodes:
                    canonical = ENode(
                        node.op,
                        tuple(self.find(child) for child in node.children),
                        node.attrs_key,
                        node.source_node,
                        node.node,
                        node.ordinal,
                        node.original,
                    )
                    key = canonical.key
                    existing = new_memo.get(key)
                    if existing is not None and self.find(existing) != root:
                        pending_unions.append((existing, root))
                    else:
                        new_memo[key] = root
                        if canonical not in rewritten:
                            rewritten.append(canonical)
                self._nodes[root] = rewritten
            self._memo = new_memo
            if not pending_unions:
                break
            for lhs, rhs in pending_unions:
                lhs = self.find(lhs)
                rhs = self.find(rhs)
                if lhs == rhs:
                    continue
                if self._types[lhs] != self._types[rhs]:
                    raise IRVerificationError("Congruence rebuild found incompatible e-class types.")
                root, merged = (lhs, rhs) if lhs < rhs else (rhs, lhs)
                self._parent[merged] = root
                self._nodes[root].extend(self._nodes[merged])
                changed = True
        if changed:
            self.version += 1

    def class_for_node(self, node_id: str) -> int:
        try:
            return self.find(self._node_class[node_id])
        except KeyError as error:
            raise IRVerificationError(
                f"E-graph expression references unknown node {node_id!r}.", node_id=node_id) from error

    def has_node_id(self, node_id: str) -> bool:
        return node_id in self._node_class

    @property
    def source_nodes(self) -> tuple[Node, ...]:
        """Named expressions, including aliases removed by congruence."""
        return tuple(self._source_nodes.values())

    def classes(self) -> tuple[EClassView, ...]:
        return tuple(
            EClassView(class_id, self._types[class_id], tuple(self._nodes[class_id]))
            for class_id in range(len(self._parent))
            if self.find(class_id) == class_id
        )

    def class_view(self, class_id: int) -> EClassView:
        """Return the canonical view for one e-class."""

        root = self.find(class_id)
        return EClassView(root, self._types[root], tuple(self._nodes[root]))

    def enodes(self) -> tuple[tuple[int, ENode], ...]:
        return tuple((view.id, node) for view in self.classes() for node in view.nodes)

    def reachable(self, roots: Iterable[int]) -> tuple[EClassView, ...]:
        pending = [self.find(value) for value in roots]
        visited: set[int] = set()
        while pending:
            class_id = self.find(pending.pop())
            if class_id in visited:
                continue
            visited.add(class_id)
            for node in self._nodes[class_id]:
                pending.extend(self.find(child) for child in node.children)
        return tuple(view for view in self.classes() if view.id in visited)

    def roots_for_module(self, module: IRModule) -> tuple[int, ...]:
        root_ids = {output for function in module.functions for output in function.outputs}
        root_ids.update(input_id for node in module.nodes if not node.effect.is_pure for input_id in node.inputs)
        roots = {
            self.class_for_node(node_id)
            for node_id in root_ids
            if self.has_node_id(node_id)
        }
        return tuple(sorted(roots))

    def to_dot(
        self,
        *,
        roots: Iterable[int] = (),
        costs: Mapping[ENode, float] | None = None,
        picks: Mapping[ENode, bool] | None = None,
    ) -> str:
        """Return a self-contained Graphviz representation."""

        root_set = {self.find(value) for value in roots}
        lines = [
            "digraph EGraph {",
            '  graph [rankdir="BT", compound="true"];',
            '  node [fontname="monospace"];',
        ]
        for view in self.classes():
            class_color = "#fff2a8" if view.id in root_set else "#d9e8fb"
            lines.extend((
                f"  subgraph cluster_{view.id} {{",
                f'    label="eclass {view.id} : {_dot_escape(str(view.type))}";',
                f'    color="{class_color}";',
                f'    c{view.id} [shape="ellipse", label="e{view.id}", style="filled", fillcolor="{class_color}"];',
            ))
            for index, node in enumerate(view.nodes):
                name = f"n{view.id}_{index}"
                picked = bool(picks and picks.get(node, False))
                fill = "#9be79b" if picked else "white"
                label = f"{node.op}\\n{node.source_node}"
                if costs is not None and node in costs:
                    label += f"\\ncost={costs[node]:.6g}"
                if picked:
                    label += "\\nPICK"
                lines.append(
                    f'    {name} [shape="box", label="{_dot_escape(label)}", style="filled", fillcolor="{fill}"];')
                lines.append(f'    {name} -> c{view.id} [style="dotted"];')
            lines.append("  }")
        for view in self.classes():
            for index, node in enumerate(view.nodes):
                name = f"n{view.id}_{index}"
                for input_index, child in enumerate(node.children):
                    lines.append(f'  {name} -> c{self.find(child)} [label="{input_index}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"


def _dot_escape(value: str) -> str:
    return html.escape(value, quote=True).replace("\n", "\\n")


def attrs_key_for(node: Node) -> str:
    """Return the deterministic semantic leaf key used for hash-consing.

    ``Node.attrs`` may contain IRType instances, which the ordinary JSON
    encoder cannot serialize.  ``Node.to_data`` already owns the canonical
    IR attribute encoding, so the e-graph must use the same representation.
    Variables remain identity-bearing leaves, matching nncase's globally
    unique ``Var`` equality rather than merging same-spelled parameters.
    """

    node_data = node.to_data()
    attrs = node_data["attrs"]
    if node.op in {"builtin.var", "egraph.opaque"}:
        attrs = {"semantic_attrs": attrs, "source_identity": node.id}
    # A typed e-node is (op, children, semantic attributes, result type).
    # Most result types are inferable from the first three fields, but source
    # operations such as splat/scalar constants intentionally omit their
    # result_type constructor argument from Node.attrs.  Leaving type out
    # would make differently shaped constants falsely congruent.
    key = {"attrs": attrs, "type": node_data["type"]}
    return json.dumps(key, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = ["EClassView", "EGraph", "ENode", "attrs_key_for"]
