# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import re
import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import il_source, script_source
from .helpers import arithmetic_module


@pytest.mark.parametrize("render", (il_source, script_source))
def test_short_ssa_on_left_original_names_on_right(render):
    module = arithmetic_module()
    before = module.semantic_hash
    source = render(module)
    assignments = [line.strip() for line in source.splitlines() if line.startswith("  %")]
    assert assignments[0].startswith("%0 = Math.Add(")
    assert assignments[1].startswith("%1 = Math.Mul(%0,")
    assert assignments[0].endswith("// f32[4] name='very_long_original_output_name'")
    assert assignments[1].endswith("// f32[4] name='out'")
    assert "(%1)" in source and "%out" not in source
    assert source == render(module) and module.semantic_hash == before


@pytest.mark.parametrize("render", (il_source, script_source))
def test_each_function_has_only_its_own_body_and_local_numbering(render):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    for name in ("main", "helper"):
        x = builder.var(name + "_input", tensor, id=name + "_input")
        y = builder.call("math.silu", (x, ), tensor, id=name + "_output")
        builder.function(name, (x, ), (y, ))
    module = fm.verify_module(builder.build(entry="main"))
    source = render(module)
    assert source.count("Math.Silu(") == 2
    assert source.count("%0 = Math.Silu(") == 2
    assert source.count("name='main_output'") == source.count("name='helper_output'") == 1


def test_numbered_parameter_cannot_collide_and_columns_stay_aligned():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="0")
    result = x
    for index in range(12):
        result = builder.call("math.silu", (result, ), tensor, id="result" + str(index))
    builder.function("main", (x, ), (result, ))
    source = il_source(fm.verify_module(builder.build(entry="main")))
    # Match ordinary and padded names without relying on a fixed digit width.
    lines = [line for line in source.splitlines() if re.match(r"  %\d+ +=", line)]
    assert len(lines) == 12
    assert len({line.index(" =") for line in lines}) == 1
    assert lines[0].lstrip().startswith("%1 ") and "Math.Silu(%0)" in lines[0]
    assert "return (%12)" in source


@pytest.mark.parametrize("render", (il_source, script_source))
def test_shared_computation_is_not_recursively_inlined(render):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="input")
    shared = builder.call("math.silu", (x, ), tensor, id="shared_value")
    result = builder.call("math.add", (shared, shared), tensor, id="result")
    builder.function("main", (x, ), (result, shared))
    source = render(fm.verify_module(builder.build(entry="main")))
    assert source.count("Math.Silu(") == 1
    assert "Math.Add(%0, %0)" in source and "(%1, %0)" in source


def test_unused_nodes_remain_visible_only_in_entry_and_effects_are_not_hidden():
    builder = fm.IRBuilder(dialect="high_level", stage="unit")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="input")
    dead = builder.call("math.silu", (x, ), tensor, id="unused")
    builder.function("main", (x, ), (x, ))
    source = il_source(builder.build(entry="main"))
    assert "name='unused'" in source and "Math.Silu(%input)" in source
    # A diagnostic printer must not hide even a malformed effectful literal.
    from triton.flagmega.ir.print_symbols import is_inline_leaf
    literal = fm.Node("effectful", "builtin.scalar_const", (), fm.tensor_type("int32", ()),
                      fm.Effect(fm.EffectKind.READ, "state"), {"value": 1})
    assert not is_inline_leaf(literal)
    assert not is_inline_leaf(dead)
