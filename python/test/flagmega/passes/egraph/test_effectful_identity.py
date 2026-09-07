# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.egraph import EGraphRewriter


def _two_effectful_results():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("bfloat16", (1, 8))
    state_type = fm.RefType("state", (("data", tensor),))
    stateful_type = fm.TupleType((tensor, state_type))
    lhs = builder.var("lhs", tensor, id="lhs")
    rhs = builder.var("rhs", tensor, id="rhs")
    state = builder.var("state", state_type, id="state")
    kernel_attrs = {
        "semantic_op": "test.effectful_boundary",
        "candidate": "reference",
        "parameters": {},
        "facts": {},
        "semantic_attrs": {},
    }
    first = builder.call(
        "tir.kernel",
        (state, lhs),
        stateful_type,
        id="first_stateful",
        effect=fm.effect("read_write", "state"),
        attrs=kernel_attrs,
    )
    first_value = builder.call(
        "builtin.get_item", (first,), tensor, id="first_value", attrs={"index": 0}
    )
    second = builder.call(
        "tir.kernel",
        (state, rhs),
        stateful_type,
        id="second_stateful",
        effect=fm.effect("read_write", "state"),
        attrs=kernel_attrs,
    )
    second_value = builder.call(
        "builtin.get_item", (second,), tensor, id="second_value", attrs={"index": 0}
    )
    builder.function("main", (lhs, rhs, state), (first_value, second_value))
    return fm.verify_module(builder.build(entry="main"))


def test_egraph_does_not_merge_distinct_effectful_boundaries():
    module = _two_effectful_results()

    result = EGraphRewriter(()).rewrite(module)

    assert result.functions[0].outputs == ("first_value", "second_value")
    assert result.node_map["first_value"].inputs == ("first_stateful",)
    assert result.node_map["second_value"].inputs == ("second_stateful",)
    effect_order = tuple(node.id for node in result.nodes if not node.effect.is_pure)
    assert effect_order == ("first_stateful", "second_stateful")
