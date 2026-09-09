# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.readonly_groups import ReadonlyGroupResolver


def _recipe(*, op="math.add", reverse_inputs=False, literal=1.0, other_key=None):
    builder = fm.IRBuilder(dialect="high_level", stage="selected_tir")
    tensor = fm.tensor_type("float32", (2, 4))
    group = {"name": "layers.weight", "index": 0, "count": 1}
    source = builder.weight("weight", tensor, source="memory", key="weight", id="weight",
                            metadata={"rdata_group": group})
    if other_key is None:
        other = builder.call("builtin.splat_const", (), tensor, attrs={"value": literal}, id="other")
    else:
        other = builder.weight(other_key, tensor, source="memory", key=other_key, id="other")
    inputs = (other, source) if reverse_inputs else (source, other)
    result = builder.call(op, inputs, tensor, id="output", metadata={"rdata_group": group})
    builder.function("main", (), (result, ))
    return freeze_constant_islands(builder.build(entry="main"))


def _name(module):
    return ReadonlyGroupResolver(module).name_for(module.node_map["output"], "layers.weight")


def test_representation_covers_op_input_order_and_literal_values():
    modules = (
        _recipe(),
        _recipe(op="math.mul"),
        _recipe(reverse_inputs=True),
        _recipe(literal=2.0),
    )
    assert len({_name(module) for module in modules}) == len(modules)


def test_ungrouped_weight_dependencies_keep_their_actual_identity():
    assert _name(_recipe(other_key="left")) != _name(_recipe(other_key="right"))


def test_independent_recipe_node_order_is_not_a_representation_contract():
    module = _recipe()
    recipe, = module.constant_recipes
    reordered = replace(module, constant_recipes=(replace(recipe,
                                                          nodes=(recipe.nodes[1], recipe.nodes[0], recipe.nodes[2])), ))
    assert _name(module) == _name(reordered)


def test_storage_type_and_distribution_participate_in_representation_identity():
    module = _recipe(other_key="other")
    recipe, = module.constant_recipes
    placement = fm.Placement((2, 2), "yx", "bb")
    names = {_name(module)}
    # Each variant is a valid homogeneous binary recipe with the same shape.
    for value_type in (
            fm.tensor_type("bfloat16", (2, 4)),
            fm.DistributedType(fm.tensor_type("float32", (2, 4)), (fm.SBP.broadcast(), fm.SBP.split_contiguous(
                (0, ), 2)), placement),
    ):
        edited = replace(
            module, nodes=tuple(replace(node, type=value_type) for node in module.nodes),
            constant_recipes=(replace(recipe, nodes=tuple(replace(node, type=value_type) for node in recipe.nodes)), ))
        names.add(_name(fm.verify_module(edited)))
    assert len(names) == 3
