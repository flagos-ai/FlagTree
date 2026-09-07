# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.passes import EGraphRulesPass, FunctionalPass, PassManager
from triton.flagmega.rules import RewriteRule


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (8,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return fm.verify_module(builder.build(entry="main"))


def _change(name: str, source_op: str, target_op: str, *, unary: bool = False) -> RewriteRule:
    return RewriteRule(
        name,
        lambda node, _module: node.op == source_op,
        lambda node, _module: replace(node, op=target_op, inputs=node.inputs[:1] if unary else node.inputs),
    )


def test_pass_manager_wraps_one_contiguous_egraph_group_once():
    manager = PassManager("rules")
    manager.add(EGraphRulesPass(
        "AddToMul",
        (_change("add-to-mul", "math.add", "math.mul"),),
        cost=lambda node, _module: {"math.add": 4.0, "math.mul": 2.0, "math.silu": 1.0}.get(node.op, 0.0),
    ))
    manager.add(EGraphRulesPass(
        "MulToSilu", (_change("mul-to-silu", "math.mul", "math.silu", unary=True),)))
    manager.add(FunctionalPass("VerifyAfterEGraph", lambda module: module))

    result = manager.run(_module())

    assert result.executed == (
        "EGraphConstructPass",
        "AddToMul",
        "MulToSilu",
        "EGraphExtractPass",
        "VerifyAfterEGraph",
    )
    assert result.module.node_map["output"].op == "math.silu"


def test_non_egraph_pass_closes_group_and_next_group_gets_new_lifecycle():
    manager = PassManager("two-groups")
    manager.add(EGraphRulesPass("First", ()))
    manager.add(FunctionalPass("Boundary", lambda module: module))
    manager.add(EGraphRulesPass("Second", ()))

    result = manager.run(_module())

    assert result.executed == (
        "EGraphConstructPass", "First", "EGraphExtractPass",
        "Boundary",
        "EGraphConstructPass", "Second", "EGraphExtractPass",
    )


def test_manager_freezes_after_automatically_closing_trailing_group():
    manager = PassManager("freeze").add(EGraphRulesPass("Rules", ()))
    manager.run(_module())
    with pytest.raises(RuntimeError, match="frozen"):
        manager.add(FunctionalPass("TooLate", lambda module: module))


def test_contiguous_egraph_passes_reject_incompatible_graph_limits():
    manager = PassManager("limits").add(EGraphRulesPass("First", (), node_limit=10))
    with pytest.raises(ValueError, match="identical graph limits"):
        manager.add(EGraphRulesPass("Second", (), node_limit=11))
