# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules import RewriteRedirect, RewriteResult, RewriteRule
from triton.flagmega.rules import DataflowRewriter


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (2, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    root = builder.call("math.add", (lhs, rhs), value_type, id="root")
    builder.function("main", (lhs, rhs), (root,))
    return fm.verify_module(builder.build(entry="main"))


def test_dataflow_pass_applies_ordered_rules_to_fixed_point():
    module = _module()
    mark = RewriteRule(
        "Mark",
        lambda node, _module: node.id == "root" and "phase" not in node.metadata,
        lambda node, _module: replace(node, metadata={"phase": 1}),
    )
    finish = RewriteRule(
        "Finish",
        lambda node, _module: node.metadata.get("phase") == 1,
        lambda node, _module: replace(node, metadata={"phase": 2}),
    )
    result = DataflowPass("Dataflow", (mark, finish)).run(module)
    assert result.node_map["root"].metadata["phase"] == 2


def test_dataflow_pass_reports_non_convergence():
    module = _module()
    toggle_add = RewriteRule(
        "AddToMul", lambda node, _module: node.op == "math.add",
        lambda node, _module: replace(node, op="math.mul"),
    )
    toggle_mul = RewriteRule(
        "MulToAdd", lambda node, _module: node.op == "math.mul",
        lambda node, _module: replace(node, op="math.add"),
    )
    with pytest.raises(IRVerificationError, match="did not converge after 2 iterations"):
        DataflowPass("Dataflow", (toggle_add, toggle_mul), max_iterations=2).run(module)


def test_dataflow_pass_rejects_helper_id_collision_and_bad_topology():
    module = _module()
    root = module.node_map["root"]
    collision = RewriteRule(
        "Collision",
        lambda node, _module: node.id == "root",
        lambda node, _module: RewriteResult(node, (replace(root, id="lhs"),)),
    )
    with pytest.raises(IRVerificationError, match="colliding helper ids"):
        DataflowPass("Dataflow", (collision,)).run(module)

    helper = fm.Node("helper", "math.silu", ("future",), root.type)
    bad_topology = RewriteRule(
        "BadTopology",
        lambda node, _module: node.id == "root",
        lambda node, _module: RewriteResult(node, (helper,)),
    )
    with pytest.raises(IRVerificationError, match="non-topological inputs"):
        DataflowPass("Dataflow", (bad_topology,)).run(module)


def test_redirect_updates_function_output_and_selection_owner():
    module = _module()
    point = fm.SelectionPoint(
        "choice.root", "test", (fm.Candidate("keep"),), "keep", owner="root",
    )
    record = fm.SelectionRecord("choice.root", "keep", "test", "test/v1")
    module = replace(module, selection_points=(point,), selections=(record,))
    redirect = RewriteRule(
        "Redirect", lambda node, _module: node.id == "root",
        lambda _node, _module: RewriteRedirect("lhs"),
    )
    result = DataflowPass("Dataflow", (redirect,)).run(module)
    assert "root" not in result.node_map
    assert result.function_map["main"].outputs == ("lhs",)
    assert result.selection_points[0].owner == "lhs"
    assert result.selection_map["choice.root"].candidate_id == "keep"


def test_redirect_must_target_a_preceding_node():
    module = _module()
    redirect = RewriteRule(
        "BadRedirect", lambda node, _module: node.id == "lhs",
        lambda _node, _module: RewriteRedirect("root"),
    )
    with pytest.raises(IRVerificationError, match="non-preceding node"):
        DataflowPass("Dataflow", (redirect,)).run(module)


def test_dataflow_rewriter_validates_iteration_budget_and_root_id():
    with pytest.raises(ValueError, match="must be positive"):
        DataflowRewriter((), max_iterations=0)

    module = _module()
    wrong_root = RewriteRule(
        "WrongRoot", lambda node, _module: node.id == "root",
        lambda node, _module: replace(node, id="other"),
    )
    with pytest.raises(IRVerificationError, match="must preserve root id"):
        DataflowPass("Dataflow", (wrong_root,)).run(module)


def test_dataflow_noop_falls_through_and_valid_helper_becomes_available():
    module = _module()
    noop = RewriteRule(
        "Noop", lambda node, _module: node.id == "root", lambda node, _module: node,
    )
    root = module.node_map["root"]
    helper = fm.Node("helper", "math.add", root.inputs, root.type)
    insert = RewriteRule(
        "Insert",
        lambda node, _module: node.id == "root" and not node.metadata,
        lambda node, _module: RewriteResult(
            replace(node, inputs=("helper", "rhs"), metadata={"rewritten": True}), (helper,),
        ),
    )
    result = DataflowPass("Dataflow", (noop, insert)).run(module)
    assert result.node_map["root"].inputs == ("helper", "rhs")
    assert result.node_map["helper"].inputs == ("lhs", "rhs")


def test_dataflow_rejects_non_topological_replacement():
    module = _module()
    bad = RewriteRule(
        "BadReplacement",
        lambda node, _module: node.id == "root",
        lambda node, _module: replace(node, inputs=("future", "rhs")),
    )
    with pytest.raises(IRVerificationError, match="replacement .* non-topological inputs"):
        DataflowPass("Dataflow", (bad,)).run(module)


def test_remove_unused_remaps_or_drops_selection_owners():
    module = _module()
    root = module.node_map["root"]
    representative = fm.Node(
        "representative", "math.silu", ("lhs",), root.type,
        metadata={"vectorization_root": "root"},
    )
    dead = fm.Node("dead", "math.silu", ("rhs",), root.type)
    function = replace(module.functions[0], outputs=("representative",))
    remapped_point = fm.SelectionPoint(
        "choice.root", "test", (fm.Candidate("keep"),), "keep", owner="root",
    )
    dropped_point = fm.SelectionPoint(
        "choice.dead", "test", (fm.Candidate("keep"),), "keep", owner="dead",
    )
    records = (
        fm.SelectionRecord("choice.root", "keep", "test", "test/v1"),
        fm.SelectionRecord("choice.dead", "keep", "test", "test/v1"),
    )
    prepared = replace(
        module,
        nodes=(*module.nodes, dead, representative),
        functions=(function,),
        selection_points=(remapped_point, dropped_point),
        selections=records,
    )
    result = DataflowPass("DCE", ()).run(prepared)
    assert "root" not in result.node_map
    assert result.selection_points == (replace(remapped_point, owner="representative"),)
    assert set(result.selection_map) == {"choice.root"}


def test_remove_unused_uses_outermost_surviving_selection_representative():
    module = _module()
    root = module.node_map["root"]
    inner = fm.Node(
        "inner", "math.silu", ("lhs",), root.type,
        metadata={"vectorization_root": "root"},
    )
    outer = fm.Node(
        "outer", "math.silu", ("inner",), root.type,
        metadata={"vectorization_root": "root"},
    )
    point = fm.SelectionPoint(
        "choice.root", "test", (fm.Candidate("keep"),), "keep", owner="root",
    )
    prepared = replace(
        module,
        nodes=(*module.nodes, inner, outer),
        functions=(replace(module.functions[0], outputs=("outer",)),),
        selection_points=(point,),
        selections=(fm.SelectionRecord("choice.root", "keep", "test", "test/v1"),),
    )

    result = DataflowPass("DCE", ()).run(prepared)

    assert result.selection_points[0].owner == "outer"
    assert result.selection_map["choice.root"].candidate_id == "keep"
