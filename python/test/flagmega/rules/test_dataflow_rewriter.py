# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
from functools import cached_property

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.rules import DataflowRewriter, RewriteRule


def test_iteration_limit_counts_fixed_point_sweeps_not_matching_nodes():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    values = tuple(
        builder.var(f"value_{index}", fm.tensor_type("float32", (1,)), id=f"value_{index}")
        for index in range(40)
    )
    builder.function("main", values, values)
    module = builder.build(entry="main")

    rule = RewriteRule(
        "mark",
        lambda node, _: not bool(node.metadata.get("marked", False)),
        lambda node, _: replace(node, metadata={**dict(node.metadata), "marked": True}),
    )
    rewritten = DataflowRewriter(
        (rule,), max_iterations=2, remove_unused=False).rewrite(module)

    assert len(rewritten.nodes) == 40
    assert all(node.metadata["marked"] is True for node in rewritten.nodes)


def test_local_edits_reuse_unchanged_node_index(monkeypatch):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    values = tuple(builder.var(f"x{i}", fm.tensor_type("float32", (1,)), id=f"x{i}") for i in range(40))
    builder.function("main", values, values)
    module = builder.build(entry="main")
    builds = []
    original = fm.IRModule.node_map.func

    def build_index(view):
        builds.append(view)
        return original(view)

    descriptor = cached_property(build_index)
    descriptor.__set_name__(fm.IRModule, "node_map")
    monkeypatch.setattr(fm.IRModule, "node_map", descriptor)

    def matches(node, view):
        assert view.node_map[node.id] is node
        return not node.metadata.get("marked", False)

    rule = RewriteRule("mark-indexed", matches,
                       lambda node, _: replace(node, metadata={"marked": True}))
    rewritten = DataflowRewriter((rule,), max_iterations=2, remove_unused=False).rewrite(module)
    assert len(builds) <= 2
    assert all(rewritten.node_map[node.id] is node for node in rewritten.nodes)
    assert all(not node.metadata for node in module.nodes)


def test_rule_dispatch_rejects_impossible_root_operators(monkeypatch):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    x = builder.var("x", fm.tensor_type("float32", (1,)), id="x")
    builder.function("main", (x,), (x,))
    module = builder.build(entry="main")
    rule = RewriteRule("silu-only", pm.F.math.is_silu(), lambda result, _: result.root)
    calls = []
    original = rule.apply

    def measure(node, view):
        calls.append(node.id)
        return original(node, view)

    monkeypatch.setattr(rule, "apply", measure)
    assert DataflowRewriter((rule,)).rewrite(module) is module
    assert calls == []
