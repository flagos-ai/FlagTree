# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands


class CountedNodes(tuple):
    def __iter__(self):
        for node in super().__iter__():
            self.visits += 1
            yield node


def test_many_independent_islands_do_not_rescan_the_whole_graph():
    builder = fm.IRBuilder(dialect="high_level", stage="packed")
    value_type = fm.tensor_type("float32", (4,))
    weights = tuple(builder.weight(f"weight{i}", value_type, source="test.safetensors", key=f"w{i}",
                                   id=f"weight{i}") for i in range(128))
    builder.function("main", (), weights)
    module = fm.verify_module(builder.build(entry="main"))
    counted = CountedNodes(module.nodes)
    counted.visits = 0
    # The frozen tuple's values are unchanged; instrument table traversal.
    object.__setattr__(module, "nodes", counted)
    frozen = freeze_constant_islands(module)
    assert tuple(recipe.outputs for recipe in frozen.constant_recipes) == tuple((value.id,) for value in weights)
    assert tuple(recipe.nodes for recipe in frozen.constant_recipes) == tuple((value,) for value in weights)
    assert counted.visits <= 20 * len(module.nodes)
