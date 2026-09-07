# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Generic TIR traversal and immutable rewriting without generated visitors."""

from __future__ import annotations

import re
from dataclasses import fields, replace

from triton.flagmega.ir.tir.base import TIRCost, TIRNode


class TIRVisitor:
    def visit(self, node: TIRNode):
        method = getattr(self, f"visit_{_snake_name(type(node).__name__)}", None)
        return self.generic_visit(node) if method is None else method(node)

    def generic_visit(self, node: TIRNode):
        for child in iter_tir_children(node):
            self.visit(child)
        return None


class TIRRewriter(TIRVisitor):
    def __init__(self) -> None:
        self.is_mutated = False

    def rewrite(self, node: TIRNode):
        return self.visit(node)

    def generic_visit(self, node: TIRNode):
        changes = {}
        for field in fields(node):
            before = getattr(node, field.name)
            after = _rewrite_value(before, self)
            if after is not before:
                changes[field.name] = after
        if not changes:
            return node
        self.is_mutated = True
        return replace(node, **changes)


class _CostVisitor(TIRVisitor):
    def __init__(self) -> None:
        self.cost = TIRCost()

    def generic_visit(self, node: TIRNode):
        self.cost = self.cost + node.local_cost
        return super().generic_visit(node)

    def visit_producer_consumer_region(self, node):
        # Producer and consumer PipelineStage nodes are two execution-role
        # views of one semantic kernel, not two kernel invocations.  Counting
        # the consumer body once includes the kernel plus each logical
        # drain/handoff exactly once.
        self.cost = self.cost + node.local_cost
        self.visit(node.consume_body)
        return None


def estimate_tir_cost(node: TIRNode) -> TIRCost:
    visitor = _CostVisitor()
    visitor.visit(node)
    return visitor.cost


def iter_tir_children(node: TIRNode):
    for field in fields(node):
        yield from _children(getattr(node, field.name))


def _children(value: object):
    if isinstance(value, TIRNode):
        yield value
    elif isinstance(value, tuple):
        for item in value:
            yield from _children(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _children(item)


def _rewrite_value(value: object, rewriter: TIRRewriter):
    if isinstance(value, TIRNode):
        return rewriter.visit(value)
    if isinstance(value, tuple):
        rewritten = tuple(_rewrite_value(item, rewriter) for item in value)
        return value if all(left is right for left, right in zip(value, rewritten)) else rewritten
    if isinstance(value, dict):
        rewritten = {key: _rewrite_value(item, rewriter) for key, item in value.items()}
        return value if all(rewritten[key] is item for key, item in value.items()) else rewritten
    return value


def _snake_name(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


__all__ = ["TIRRewriter", "TIRVisitor", "estimate_tir_cost", "iter_tir_children"]
