# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.ir.printer import il_source, script_source
from triton.flagmega.passes import freeze_constant_islands


def weight_chain_module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="input")
    weight = builder.weight("weight", tensor, source="missing", key="weight", id="weight")
    one = builder.node(op="builtin.splat_const", type=tensor, attrs={"value": 1.0}, id="one")
    transformed = builder.call("math.add", (weight, one), tensor, id="weight_add")
    transformed = builder.call("math.silu", (transformed, ), tensor, id="weight_silu")
    output = builder.call("math.add", (x, transformed), tensor, id="output")
    builder.function("main", (x, ), (output, transformed))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("render", (il_source, script_source))
def test_weight_expressions_are_in_a_function_local_section(render):
    module = weight_chain_module()
    before = fm.module_source(module)
    source = render(module)
    weights, compute = source.split("  // compute\n")
    assert "  weights {\n" in weights
    assert "    %w0 = Math.Add(" in weights and "    %w1 = Math.Silu(%w0)" in weights
    assert "name='weight_add'" in weights and "name='weight_silu'" in weights
    assert "  %0 = Math.Add(%input, %w1)" in compute
    assert "(%0, %w1)" in compute
    assert "name='weight_add'" not in compute and "WeightRef(" not in compute
    assert source == render(module) and fm.module_source(module) == before


@pytest.mark.parametrize("render", (il_source, script_source))
def test_frozen_recipes_are_inside_the_function_weights_section(render):
    module = freeze_constant_islands(weight_chain_module())
    source = render(module)
    weights, compute = source.split("  // compute\n")
    assert weights.index("main") < weights.index("  weights {") < weights.index("fingerprint=")
    assert "name='weight_silu'" in weights and "name='weight_silu'" not in compute
    assert "ConstAssetRef(" in compute


@pytest.mark.parametrize("render", (il_source, script_source))
def test_distributed_constant_transforms_separate_from_runtime_boxing(render):
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    tensor = fm.tensor_type("float32", (4, ))
    dist = fm.DistributedType(tensor, (fm.SBP.split_contiguous((0, )), ), fm.Placement((2, ), "x", "b"))
    weight = builder.weight("w", tensor, source="missing", key="w", id="w")
    x = builder.var("x", tensor, id="x")
    results = []
    for value in (weight, x):
        shard = builder.call("distributed.sharded_view", (value, ), dist, attrs={"new_type": dist},
                             id=value.id + "_shard")
        boxed = builder.call("distributed.boxing", (shard, ), tensor, attrs={"new_type": tensor},
                             id=value.id + "_boxing")
        results.append(boxed)
    builder.function("main", (x, ), results)
    source = render(fm.verify_module(builder.build(entry="main")))
    weights, compute = source.split("  // compute\n")
    assert "name='w_shard'" in weights and "name='w_boxing'" in weights
    assert "name='x_shard'" in compute and "name='x_boxing'" in compute
    assert "[2@x]" in weights


def test_weight_numbering_does_not_collide_with_parameter_names():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (4, ))
    x = builder.var("input", tensor, id="w0")
    weight = builder.weight("w", tensor, source="missing", key="w", id="w")
    transformed = builder.call("math.silu", (weight, ), tensor, id="transform")
    output = builder.call("math.add", (x, transformed), tensor, id="out")
    builder.function("main", (x, ), (output, ))
    source = il_source(fm.verify_module(builder.build(entry="main")))
    assert "%w1 = Math.Silu(" in source
    assert "%0 = Math.Add(%w0, %w1)" in source
