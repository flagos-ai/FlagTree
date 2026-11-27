# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm


def _module_with_dead_node() -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", [1, 8])
    source = builder.var("source", tensor, id="source")
    live = builder.call("math.silu", [source], tensor, id="live")
    builder.call("math.add", [source, source], tensor, id="dead")
    builder.function("main", [source], [live])
    return builder.build(entry="main")


def test_module_match_walks_outputs_before_operands_and_ignores_dead_nodes():
    module = _module_with_dead_node()

    match = pm.try_match(module, pm.wildcard("root"))

    assert match is not None
    assert match["root"].id == "live"
    assert pm.try_match(module, pm.F.math.is_add()) is None
    assert pm.find_matches(module, pm.F.math.is_add()) == ()
    assert pm.try_match_root(
        module.node_map["dead"], pm.F.math.is_add(), module
    ) is not None


def test_match_options_suppress_by_node_identity_and_can_be_inherited():
    module = _module_with_dead_node()
    pattern = pm.F.math.is_silu(call_name="activation")
    options = pm.MatchOptions()
    source = module.node_map["live"]
    destination = replace(source, id="replacement")

    assert pm.try_match_root(source, pattern, module, options) is not None
    options.suppress(source, pattern)
    assert pm.try_match_root(source, pattern, module, options) is None

    options.inherit(source, destination)
    assert options.is_suppressed(destination, pattern)

    # A structurally equal id in another module is a different expression.
    other = _module_with_dead_node().node_map["live"]
    assert not options.is_suppressed(other, pattern)


def test_pattern_user_count_includes_distinct_ssa_and_function_output_users():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", [1, 8])
    source = builder.var("source", tensor, id="source")
    activated = builder.call("math.silu", [source], tensor, id="activated")
    lhs = builder.call("math.add", [activated, source], tensor, id="lhs")
    rhs = builder.call("math.mul", [activated, source], tensor, id="rhs")
    builder.function("main", [source], [lhs, rhs])
    module = builder.build(entry="main")

    two_users = pm.F.math.is_silu(call_name="activation").with_user_count(2)
    one_user = pm.F.math.is_silu().with_user_count(1)

    assert pm.try_match_root(activated, two_users, module)["activation"] is activated
    assert pm.try_match_root(activated, one_user, module) is None
    # Function outputs participate in the nncase-style Users relation.
    assert pm.try_match_root(lhs, pm.F.math.is_add().with_user_count(1), module)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "1"])
def test_pattern_user_count_rejects_non_count_values(invalid):
    with pytest.raises(ValueError, match="non-negative integer"):
        pm.wildcard().with_user_count(invalid)


def test_const_pattern_supports_exact_predicate_and_type_constraints():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    scalar_type = fm.tensor_type("int32", [])
    constant = builder.call(
        "builtin.scalar_const", (), scalar_type, id="answer", attrs={"value": 42}
    )
    builder.function("main", (), (constant,))
    module = builder.build(entry="main")

    exact = pm.is_const(42, name="constant", type_pattern=fm.has_dtype("int32"))
    predicate = pm.is_const(condition=lambda value: value % 7 == 0)

    assert pm.try_match_root(constant, exact, module)["constant"] is constant
    assert pm.try_match_root(constant, predicate, module)
    assert pm.try_match_root(constant, pm.is_const(41), module) is None
    assert pm.try_match_root(
        constant, pm.is_const(type_pattern=fm.has_dtype("float32")), module
    ) is None
