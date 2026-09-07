# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes import EGraphRulesPass
from triton.flagmega.rules import RewriteResult, RewriteRule


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (2, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    root = builder.call("math.add", (lhs, rhs), value_type, id="root")
    builder.function("main", (lhs, rhs), (root,))
    return fm.verify_module(builder.build(entry="main"))


def _commute_rule():
    return RewriteRule(
        "Commute",
        lambda node, _module: node.op == "math.add",
        lambda node, _module: replace(node, inputs=tuple(reversed(node.inputs))),
    )


def test_default_extraction_keeps_original_on_equal_cost():
    module = _module()
    assert EGraphRulesPass("EGraph", (_commute_rule(),)).run(module) == module


def test_cost_and_selector_can_extract_a_registered_alternative():
    module = _module()
    by_cost = EGraphRulesPass(
        "EGraph",
        (_commute_rule(),),
        cost=lambda node, _module: 0.0 if node.inputs == ("rhs", "lhs") else 1.0,
    ).run(module)
    assert by_cost.node_map["root"].inputs == ("rhs", "lhs")

    by_selector = EGraphRulesPass(
        "EGraph",
        (_commute_rule(),),
        selector=lambda _original, alternatives, _module: alternatives[0].node,
    ).run(module)
    assert by_selector.node_map["root"].inputs == ("rhs", "lhs")


def test_egraph_rule_rejects_type_change_before_extraction():
    module = _module()
    rule = RewriteRule(
        "ChangeType",
        lambda node, _module: node.id == "root",
        lambda node, _module: replace(node, type=fm.tensor_type("float32", (2, 16))),
    )
    with pytest.raises(IRVerificationError, match="preserve type and purity"):
        EGraphRulesPass("EGraph", (rule,)).run(module)


def test_egraph_rule_rejects_effectful_and_non_topological_helpers():
    module = _module()
    root = module.node_map["root"]
    effectful = fm.Node(
        "helper", "math.add", root.inputs, root.type, fm.effect("write", "state"),
    )
    effect_rule = RewriteRule(
        "EffectfulHelper",
        lambda node, _module: node.id == "root",
        lambda node, _module: RewriteResult(replace(node, inputs=("helper", "rhs")), (effectful,)),
    )
    with pytest.raises(IRVerificationError, match="cannot insert effectful helper"):
        EGraphRulesPass("EGraph", (effect_rule,)).run(module)

    non_topological = fm.Node("helper", "math.silu", ("future",), root.type)
    topology_rule = RewriteRule(
        "NonTopologicalHelper",
        lambda node, _module: node.id == "root",
        lambda node, _module: RewriteResult(node, (non_topological,)),
    )
    with pytest.raises(IRVerificationError, match="non-topological inputs"):
        EGraphRulesPass("EGraph", (topology_rule,)).run(module)


def test_egraph_selector_must_return_a_known_alternative():
    module = _module()
    with pytest.raises(IRVerificationError, match="not one of the alternatives"):
        EGraphRulesPass(
            "EGraph",
            (_commute_rule(),),
            selector=lambda original, _alternatives, _module: replace(original, op="math.mul"),
        ).run(module)


def test_egraph_rule_and_selector_normalize_root_ids():
    module = _module()
    bad_root = RewriteRule(
        "BadRoot",
        lambda node, _module: node.id == "root",
        lambda node, _module: replace(node, id="other"),
    )
    with pytest.raises(IRVerificationError, match="preserve root id"):
        EGraphRulesPass("EGraph", (bad_root,)).run(module)

    result = EGraphRulesPass(
        "EGraph",
        (_commute_rule(),),
        selector=lambda _original, alternatives, _module: replace(alternatives[0].node, id="temporary"),
    ).run(module)
    assert result.node_map["root"].inputs == ("rhs", "lhs")


def test_egraph_extraction_rejects_helper_collision_and_bad_replacement_topology():
    module = _module()
    root = module.node_map["root"]
    colliding = replace(root, id="lhs")
    collision_rule = RewriteRule(
        "Collision",
        lambda node, _module: node.id == "root",
        lambda node, _module: RewriteResult(replace(node, op="math.mul"), (colliding,)),
    )
    with pytest.raises(IRVerificationError, match="colliding helper ids"):
        EGraphRulesPass(
            "EGraph", (collision_rule,),
            selector=lambda _original, alternatives, _module: alternatives[0].node,
        ).run(module)

    topology_rule = RewriteRule(
        "BadReplacementTopology",
        lambda node, _module: node.id == "root",
        lambda node, _module: replace(node, inputs=("future", "rhs")),
    )
    with pytest.raises(IRVerificationError, match="replacement .* non-topological inputs"):
        EGraphRulesPass("EGraph", (topology_rule,)).run(module)
