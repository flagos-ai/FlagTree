# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator.numpy_backend import iter_numpy_materialized_constant_assets
from triton.flagmega.evaluator.torch_backend import (
    CheckpointWeightResolver, iter_materialized_constant_assets, materialize_constant_assets,
)
from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
from triton.flagmega.passes.constants import freeze_constant_islands


class CountedNodes(tuple):
    visits = 0

    def __iter__(self):
        for value in super().__iter__():
            type(self).visits += 1
            yield value


def test_materialization_type_index_visits_each_recipe_once():
    builder = fm.IRBuilder(dialect="ntt", stage="canonical_constants")
    value_type = fm.tensor_type("float32", (8,))
    weights = tuple(builder.weight(f"w{i}", value_type, source="memory", key=f"w{i}", id=f"w{i}")
                    for i in range(128))
    builder.function("main", (), weights)
    module = freeze_constant_islands(builder.build(entry="main"))
    source = MemoryCheckpoint({}, {f"w{i}": TensorInfo(f"w{i}", fm.DType.FLOAT32, (8,), "memory")
                                    for i in range(128)},
                              {f"w{i}": torch.full((8,), float(i)) for i in range(128)})
    for recipe in module.constant_recipes:
        object.__setattr__(recipe, "nodes", CountedNodes(recipe.nodes))
    CountedNodes.visits = 0
    outputs = dict(iter_numpy_materialized_constant_assets(module, source))
    assert len(outputs) == 128
    for i in range(128):
        np.testing.assert_array_equal(outputs[f"w{i}"], np.full((8,), i, dtype=np.float32))
    assert CountedNodes.visits <= 5 * 128


@pytest.mark.parametrize("backend", ["numpy", "torch_iterator", "torch_mapping"])
def test_closed_recipe_type_names_do_not_shadow_other_recipes(backend):
    recipes = []
    assets = []
    for index, shape in enumerate(((2, 4), (8,))):
        builder = fm.IRBuilder(dialect="ntt", stage="canonical_constants")
        weight = builder.weight("weight", fm.tensor_type("float32", shape), source="memory",
                                key=f"w{index}", id="weight")
        output_type = fm.tensor_type(fm.vector_type("float32", (2,)),
                                     (2, 2) if index == 0 else (4,))
        output = builder.call("tensors.pack", (weight,), output_type, id=f"out{index}",
                              attrs={"lanes": (2,), "axes": (len(shape) - 1,)})
        builder.function("main", (), (output,))
        recipe = fm.ConstantRecipe(f"recipe{index}", builder.build(entry="main").nodes, (output.id,))
        recipes.append(recipe)
        assets.append(fm.Node(output.id, "builtin.const_asset", (), output.type,
                              attrs={"recipe": recipe.id, "output": output.id}))
    module = fm.verify_module(fm.IRModule("ntt", "constants_frozen", tuple(assets),
                                         (fm.Function("main", (), tuple(value.id for value in assets)),), "main",
                                         metadata={"constant_phase": "frozen"},
                                         constant_recipes=tuple(recipes)))
    source = MemoryCheckpoint({}, {f"w{i}": TensorInfo(f"w{i}", fm.DType.FLOAT32, shape, "memory")
                                    for i, shape in enumerate(((2, 4), (8,)))},
                              {"w0": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                               "w1": torch.arange(8, dtype=torch.float32)})
    if backend == "numpy":
        outputs = dict(iter_numpy_materialized_constant_assets(module, source))
    elif backend == "torch_iterator":
        outputs = dict(iter_materialized_constant_assets(module, CheckpointWeightResolver(source, cache=False)))
    else:
        outputs = materialize_constant_assets(module, CheckpointWeightResolver(source, cache=False))
    np.testing.assert_array_equal(outputs["out0"], np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    np.testing.assert_array_equal(outputs["out1"], np.arange(8, dtype=np.float32).reshape(4, 2))
