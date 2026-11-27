# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.egraph import EGraphRewriter
from triton.flagmega.rules import RewriteRule


def _nested_module() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (8,))
    x = builder.var("x", tensor, id="x")
    y = builder.var("y", tensor, id="y")
    activated = builder.call("math.silu", (x,), tensor, id="activated")
    root = builder.call("math.add", (activated, y), tensor, id="root")
    builder.function("main", (x, y), (root,))
    return fm.verify_module(builder.build(entry="main"))


def test_rule_matches_alternative_inside_child_eclass_on_later_iteration():
    x = pm.wildcard("x")
    expose_mul = RewriteRule(
        "silu-to-self-mul",
        pm.F.math.is_silu(x),
        lambda match, _module: replace(
            match.root,
            op="math.mul",
            inputs=(match["x"].id, match["x"].id),
            metadata={"egraph_step": 1},
        ),
    )
    nested_x = pm.wildcard("nested_x")
    y = pm.wildcard("y")
    consume_virtual_mul = RewriteRule(
        "add-of-virtual-mul",
        pm.F.math.is_add(pm.F.math.is_mul(nested_x, nested_x), y),
        lambda match, _module: replace(
            match.root,
            op="math.mul",
            inputs=(match["nested_x"].id, match["y"].id),
            metadata={"egraph_step": 2},
        ),
    )

    rewriter = EGraphRewriter(
        (expose_mul, consume_virtual_mul),
        cost=lambda node, _module: 0.0 if node.metadata.get("egraph_step") == 2 else 10.0,
    )
    result = rewriter.rewrite(_nested_module())

    assert result.node_map["root"].op == "math.mul"
    assert result.node_map["root"].inputs == ("x", "y")
    assert result.node_map["root"].metadata["egraph_step"] == 2


def test_repeated_pattern_capture_uses_expression_identity_not_stable_id():
    x = pm.wildcard("x")
    add_same = RewriteRule(
        "same-operands-only",
        pm.F.math.is_add(x, x),
        lambda match, _module: replace(match.root, op="math.mul"),
    )

    result = EGraphRewriter((add_same,)).rewrite(_nested_module())

    assert result.node_map["root"].op == "math.add"
