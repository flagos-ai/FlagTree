# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes import DataflowPass, EGraphRulesPass, PassManager
from triton.flagmega.rules import RewriteRule


def _add_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("bfloat16", (1, 16))
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", [lhs, rhs], value_type, id="output")
    builder.function("main", [lhs, rhs], [output])
    return builder.build(entry="main")


def test_dataflow_and_egraph_rules_are_pass_manager_native():
    module = _add_module()
    dataflow_rule = RewriteRule(
        "add-to-mul",
        lambda node, _module: node.op == "math.add",
        lambda node, _module: replace(node, op="math.mul"),
    )
    dataflow = PassManager("dataflow").add(DataflowPass("Dataflow", (dataflow_rule,))).run(module).module
    assert dataflow.node_map["output"].op == "math.mul"

    swap_rule = RewriteRule(
        "commute-add",
        lambda node, _module: node.op == "math.add",
        lambda node, _module: replace(node, inputs=tuple(reversed(node.inputs))),
    )
    egraph = PassManager("egraph").add(EGraphRulesPass(
        "EGraph", (swap_rule,), selector=lambda _original, alternatives, _module: alternatives[0].node,
    )).run(module).module
    assert egraph.node_map["output"].inputs == ("rhs", "lhs")
