# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.egraph import EGraphRewriter
from triton.flagmega.rules import RewriteResult, RewriteRule


def _shared_candidate_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", (16,))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    first = builder.call("math.add", (lhs, rhs), value_type, id="first")
    second = builder.call("math.mul", (lhs, rhs), value_type, id="second")
    builder.function("main", (lhs, rhs), (first, second))
    return fm.verify_module(builder.build(entry="main"))


def _shared_candidate(node, _module):
    if node.id not in {"first", "second"} or node.metadata.get("candidate"):
        return None
    helper_id = f"shared_for_{node.id}"
    helper = fm.Node(helper_id, "math.silu", ("lhs",), node.type)
    replacement_op = "math.mul" if node.id == "first" else "math.add"
    replacement = replace(
        node,
        op=replacement_op,
        inputs=(helper_id, "rhs"),
        metadata={"candidate": True},
    )
    return RewriteResult(replacement, (helper,))


def _cost(node, _module):
    if node.op == "math.silu":
        return 12.0
    if node.metadata.get("candidate"):
        return 0.0
    if node.id in {"first", "second"}:
        return 10.0
    return 0.0


def test_cp_sat_extracts_a_globally_shared_candidate_dag():
    rewriter = EGraphRewriter((RewriteRule(
        "share-expensive-helper",
        lambda node, module: _shared_candidate(node, module) is not None,
        _shared_candidate,
    ),), cost=_cost)

    result = rewriter.rewrite(_shared_candidate_module())

    assert rewriter.last_extraction is not None
    assert rewriter.last_extraction.status in {"OPTIMAL", "FEASIBLE"}
    assert rewriter.last_extraction.objective == 12.0
    assert result.node_map["first"].metadata["candidate"] is True
    assert result.node_map["second"].metadata["candidate"] is True
    assert sum(node.op == "math.silu" for node in result.nodes) == 1


def test_equal_cost_extraction_is_deterministic_and_prefers_original():
    module = _shared_candidate_module()
    commute = RewriteRule(
        "commute",
        lambda node, _module: node.id == "first",
        lambda node, _module: replace(node, inputs=tuple(reversed(node.inputs))),
    )

    first = EGraphRewriter((commute,)).rewrite(module)
    second = EGraphRewriter((commute,)).rewrite(module)

    assert first.semantic_hash == second.semantic_hash == module.semantic_hash


def test_extraction_preserves_effectful_boundary_outside_egraph():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("bfloat16", (1, 8))
    state_type = fm.RefType("state", (("data", tensor),))
    source = builder.var("source", tensor, id="source")
    state = builder.var("state", state_type, id="state")
    stateful_type = fm.TupleType((tensor, state_type))
    stateful = builder.call(
        "tir.kernel",
        (state, source),
        stateful_type,
        id="stateful",
        effect=fm.effect("read_write", "gdn_state"),
        attrs={
            "semantic_op": "test.effectful_boundary",
            "candidate": "reference",
            "parameters": {},
            "facts": {},
            "semantic_attrs": {},
        },
    )
    value = builder.call("builtin.get_item", (stateful,), tensor, id="value", attrs={"index": 0})
    output = builder.call("math.silu", (value,), tensor, id="output")
    builder.function("main", (source, state), (output,))
    module = fm.verify_module(builder.build(entry="main"))

    result = EGraphRewriter(()).rewrite(module)

    assert result.semantic_hash == module.semantic_hash
    assert result.node_map["stateful"].effect == module.node_map["stateful"].effect
