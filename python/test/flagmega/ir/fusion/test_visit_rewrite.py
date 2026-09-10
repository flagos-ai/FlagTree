# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.ir.visitor import IRVisitor
from triton.flagmega.egraph import EGraphRewriter
from triton.flagmega.pattern_match import F, wildcard
from triton.flagmega.rules import DataflowRewriter, RewriteRule, RewriteRedirect
from python.test.flagmega.ir.fusion.test_pre_post_ops import make_graph


def test_visitor_enters_semantic_bodies_without_attribute_flag():
    calls, bodies = [], []

    class Visitor(IRVisitor):

        def visit_tensors_cast(self, node):
            calls.append(node.id)

        def visit_fusion(self, body):
            bodies.append(body.name)

    Visitor().run(make_graph())
    assert calls == ["wide", "narrow"]
    assert bodies == ["widen", "narrow"]


def test_naked_op_rule_cannot_drop_attached_fusions():
    # A rule that would be valid for identity-shaped bare calls must opt in
    # before it can discard the semantics of a fused call or captured child.
    source = make_graph()
    rule = RewriteRule("NakedSoftmax", F.nn.is_softmax(wildcard("x")),
                       lambda match, module: RewriteRedirect(match["x"].id))
    assert DataflowRewriter((rule, )).rewrite(source).semantic_hash == source.semantic_hash
    assert EGraphRewriter((rule, )).rewrite(source).node_map["y"].attrs == source.node_map["y"].attrs


def test_fusion_survives_empty_egraph_round_trip():
    source = make_graph()
    result = EGraphRewriter(()).rewrite(source)
    assert result.node_map["y"].attrs == source.node_map["y"].attrs


def test_cost_reports_external_storage_without_hidden_cast_buffers():
    source = make_graph()
    metric = fm.get_cost(source.node_map["y"])
    assert metric.bytes_written == 2 * 16 * 2
    assert "pre-post-fusion" in metric.notes


def test_candidate_cost_accounts_for_real_boundary_storage():
    source = make_graph()
    node = source.node_map["y"]
    metric = fm.get_definition(node.op).cost_factors(tuple(source.node_map[value] for value in node.inputs), node.attrs,
                                                     node.type)
    assert metric.block_local_memory_load_bytes == 2 * 16 * 2
    assert metric.block_local_memory_store_bytes == 2 * 16 * 2


def test_evaluator_coverage_enters_fusion_bodies(monkeypatch):
    from triton.flagmega.evaluator import inspect_evaluation_support
    from triton.flagmega.ir.ops.core import OpDefinition
    source = make_graph()
    monkeypatch.setattr(fm.get_definition("tensors.cast"), "evaluate", OpDefinition.evaluate)
    support = inspect_evaluation_support(source)
    assert not support.is_complete
    assert {gap.node_id for gap in support.gaps} == {"wide", "narrow"}
