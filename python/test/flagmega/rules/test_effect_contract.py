# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.rules import (
    DataflowRewriter,
    RewriteEffectPolicy,
    RewriteRedirect,
    RewriteResult,
    RewriteRule,
)


def _effectful_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", [2])
    source = builder.var("source", value_type, id="source")
    stateful = builder.call(
        "tir.kernel",
        (source,),
        value_type,
        id="stateful",
        effect=fm.effect("read_write", "state"),
        attrs={
            "semantic_op": "test.stateful",
            "candidate": "unit",
            "parameters": {},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    builder.function("main", (source,), (stateful,))
    return fm.verify_module(builder.build(entry="main"))


def test_redirect_cannot_silently_delete_an_effectful_node():
    module = _effectful_module()
    rule = RewriteRule(
        "DeleteEffect",
        lambda node, _module: node.id == "stateful",
        lambda _node, _module: RewriteRedirect("source"),
    )

    with pytest.raises(IRVerificationError, match="cannot redirect effectful node"):
        DataflowRewriter((rule,)).rewrite(module)


def test_replacement_and_prefix_helpers_preserve_effects_by_default():
    module = _effectful_module()
    root = module.node_map["stateful"]
    drop_effect = RewriteRule(
        "DropEffect",
        lambda node, _module: node.id == "stateful",
        lambda node, _module: replace(node, effect=fm.PURE),
    )
    with pytest.raises(IRVerificationError, match="must preserve root effect"):
        DataflowRewriter((drop_effect,)).rewrite(module)

    effectful_helper = replace(root, id="helper")
    insert_effect = RewriteRule(
        "InsertEffect",
        lambda node, _module: node.id == "stateful",
        lambda node, _module: RewriteResult(node, (effectful_helper,)),
    )
    with pytest.raises(IRVerificationError, match="inserted effectful helpers"):
        DataflowRewriter((insert_effect,)).rewrite(module)


def test_effect_change_requires_an_explicit_rule_contract():
    module = _effectful_module()
    rule = RewriteRule(
        "ExplicitlyDeleteEffect",
        lambda node, _module: node.id == "stateful",
        lambda _node, _module: RewriteRedirect("source"),
        effect_policy=RewriteEffectPolicy.ALLOW,
    )

    result = DataflowRewriter((rule,)).rewrite(module)
    assert "stateful" not in result.node_map
    assert result.function_map["main"].outputs == ("source",)


def test_redirect_must_preserve_type_even_for_pure_nodes():
    module = _effectful_module()
    wrong = fm.Node("wrong", "builtin.scalar_const", (), fm.tensor_type("int32", []), attrs={"value": 1})
    prepared = replace(module, nodes=(wrong, *module.nodes))
    rule = RewriteRule(
        "WrongType",
        lambda node, _module: node.id == "source",
        lambda _node, _module: RewriteRedirect("wrong"),
    )
    with pytest.raises(IRVerificationError, match="redirect must preserve root type"):
        DataflowRewriter((rule,)).rewrite(prepared)
