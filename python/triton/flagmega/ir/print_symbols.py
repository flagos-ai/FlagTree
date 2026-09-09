# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Display-only graph scopes and short SSA names, independent of IR identity."""

from collections.abc import Callable, Sequence, Set

from triton.flagmega.ir.model import IRModule, Node

_INLINE_LEAVES = frozenset({
    "builtin.weight",
    "builtin.const_asset",
    "builtin.scalar_const",
    "builtin.splat_const",
    "tir.scalar_const",
})


def is_inline_leaf(node: Node) -> bool:
    """Inline references/literals, never their computations or effectful nodes."""
    return not node.inputs and node.effect.is_pure and (node.op in _INLINE_LEAVES or
                                                        (node.op == "tir.buffer"
                                                         and node.attrs.get("storage") == "rdata"))


class GraphPrintSymbols:
    """One function/recipe's display names; never rename the underlying nodes."""

    def __init__(self, nodes: Sequence[Node], render_leaf: Callable[[Node], str],
                 weight_values: Set[str] = frozenset()) -> None:
        self.nodes = tuple(node for node in nodes if node.op != "builtin.var" and not is_inline_leaf(node))
        self.weights = tuple(node for node in self.nodes if node.id in weight_values)
        self.body = tuple(node for node in self.nodes if node.id not in weight_values)
        self._references = {node.id: f"%{node.id}" for node in nodes if node.op == "builtin.var"}
        reserved = set(self._references.values())
        self._widths = {}
        for prefix, group in (("%w", self.weights), ("%", self.body)):
            number = 0
            for node in group:
                while f"{prefix}{number}" in reserved:
                    number += 1
                self._references[node.id] = f"{prefix}{number}"
                number += 1
            width = max((len(self._references[node.id]) for node in group), default=0)
            self._widths.update((node.id, width) for node in group)
        for node in nodes:
            if is_inline_leaf(node):
                self._references[node.id] = render_leaf(node)

    def reference(self, node_id: str) -> str:
        return self._references[node_id]

    def lhs(self, node: Node) -> str:
        return self.reference(node.id).ljust(self._widths[node.id])


def function_node_scopes(module: IRModule) -> dict[str, tuple[Node, ...]]:
    """Match function dump closures without constructing/copying module metadata.

    Keep unreferenced nodes in the entry's diagnostic body, as function_module's
    include_unreferenced dump mode does. Do not repeat another callee's body.
    """
    node_map = module.node_map
    scopes = {}
    referenced = set()
    for function in module.functions:
        reachable = set(function.parameters)
        pending = list(function.outputs)
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            pending.extend(node_map[node_id].inputs)
        scopes[function.name] = reachable
        referenced.update(reachable)
    if module.entry in scopes:
        scopes[module.entry].update(node_map.keys() - referenced)
    return {name: tuple(node for node in module.nodes if node.id in reachable) for name, reachable in scopes.items()}
