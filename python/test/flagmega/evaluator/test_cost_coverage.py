# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import inspect_cost_coverage


def _add_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", [8])
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    output = builder.call("math.add", (lhs, rhs), value_type, id="output")
    builder.function("main", (lhs, rhs), (output,))
    return fm.verify_module(builder.build(entry="main"))


def test_unknown_cost_factors_are_not_encoded_as_zero():
    cost = fm.OpCost(notes=("not-modeled",))

    assert cost.kind is fm.CostKind.UNKNOWN
    assert cost.flops is None
    assert set(cost.unknown_factors) == {
        "flops",
        "bytes_read",
        "bytes_written",
        "communication_bytes",
        "synchronizations",
    }
    assert not cost.is_complete


def test_known_local_factors_remain_partial_until_every_factor_is_proven():
    report = inspect_cost_coverage(_add_module())
    output = next(value for value in report.entries if value.node_id == "output")

    assert output.cost.kind is fm.CostKind.ANALYTIC
    assert output.cost.flops == 8
    assert output.cost.bytes_read == 64
    assert output.cost.bytes_written == 32
    assert output.cost.unknown_factors == ("communication_bytes", "synchronizations")
    assert output in report.partial


def test_exact_zero_requires_an_explicit_contract():
    cost = fm.OpCost.exact_zero(notes=("alias-only",))

    assert cost.kind is fm.CostKind.EXACT
    assert cost.is_complete
    assert cost.flops == cost.bytes_read == cost.bytes_written == 0
