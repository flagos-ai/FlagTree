# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes import ConstnessAnalysis


def _mixed_module():
    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir_variants")
    value_type = fm.tensor_type("float32", (2, 8))
    runtime = builder.var("runtime", value_type, id="runtime")
    lhs = builder.weight("lhs", value_type, source="memory", key="lhs", id="lhs")
    rhs = builder.weight("rhs", value_type, source="memory", key="rhs", id="rhs")
    summed = builder.call("math.add", (lhs, rhs), value_type, id="weight_sum")
    activated = builder.call("math.silu", (summed,), value_type, id="weight_silu")
    output = builder.call("math.add", (runtime, activated), value_type, id="output")
    builder.function("main", (runtime,), (output,))
    return builder.build(entry="main")


def test_constness_is_derived_through_normal_ops_without_mutating_ir():
    module = _mixed_module()
    before = module.semantic_hash

    result = ConstnessAnalysis.analyze(module)

    assert result.constants == frozenset({"lhs", "rhs", "weight_sum", "weight_silu"})
    assert not result.is_constant("runtime")
    assert not result.is_constant("output")
    assert module.semantic_hash == before
    assert all("is_const" not in node.metadata for node in module.nodes)


def test_runtime_input_prevents_a_pure_const_evaluable_op_from_being_constant():
    module = _mixed_module()
    result = ConstnessAnalysis.analyze(module)

    assert module.node_map["output"].effect.is_pure
    assert fm.get_definition("math.add").const_evaluable
    assert "output" not in result.constants
