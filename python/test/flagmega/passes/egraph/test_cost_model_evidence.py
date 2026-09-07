# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.egraph import EGraphRewriter


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value_type = fm.tensor_type("float32", [4])
    value = builder.var("value", value_type, id="value")
    output = builder.call("math.silu", (value,), value_type, id="output")
    builder.function("main", (value,), (output,))
    return fm.verify_module(builder.build(entry="main"))


def test_default_extractor_names_its_structural_objective():
    rewriter = EGraphRewriter(())
    rewriter.rewrite(_module())

    assert rewriter.last_extraction is not None
    assert rewriter.last_extraction.cost_model == "structural-unit/v1"


def test_custom_callable_can_carry_a_versioned_model_identifier():
    rewriter = EGraphRewriter(
        (),
        cost=lambda _node, _module: 2.0,
        cost_model="unit-test-score/v3",
    )
    rewriter.rewrite(_module())

    assert rewriter.last_extraction is not None
    assert rewriter.last_extraction.cost_model == "unit-test-score/v3"
