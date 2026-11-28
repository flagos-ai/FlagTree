# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.errors import IRSchemaError


def concat_module(shapes, *, axis=-1, vector=False, split=False):

    class Graph(fm.Module):

        def forward(self):
            inputs = []
            for index, shape in enumerate(shapes):
                dtype = fm.vector_type("bfloat16", (2, 4)) if vector else "float32"
                value_type = fm.tensor_type(dtype, shape)
                if split:
                    policies = (fm.SBP.split_block_cyclic((0, ), 2), ) + (fm.SBP.broadcast(), ) * (len(shape) - 1)
                    value_type = fm.DistributedType(value_type, policies, fm.Placement((2, 2), "xy", "bb"))
                inputs.append(self.input(f"input_{index}", value_type, id=f"input_{index}"))
            result = fm.F.tensors.concat(*inputs, axis=axis, name="output")
            self.function("main", inputs, (result, ))

    return Graph(dialect="high_level", stage="frozen_constants", entry="main").build()


def test_concat_negative_axis_counts_logical_axes_not_vector_lane_axes():
    module = concat_module(((2, 3), (2, 5)), vector=True)
    inputs = (torch.arange(48).reshape(2, 3, 2, 4).bfloat16(), torch.arange(80).reshape(2, 5, 2, 4).bfloat16())
    actual = TorchEvaluator(DictWeightResolver({})).run(module, dict(zip(("input_0", "input_1"), inputs)))[0]
    torch.testing.assert_close(actual, torch.cat(inputs, dim=1), rtol=0, atol=0)


def test_concat_preserves_unchanged_axis_block_cyclic_distribution():
    module = concat_module(((8, 3), (8, 5)), split=True)
    result = module.node_map["output"].type
    assert isinstance(result, fm.DistributedType)
    assert result.axis_policies == module.node_map["input_0"].type.axis_policies
    assert result.tensor == fm.tensor_type("float32", (8, 8))


def test_concat_split_axis_requires_explicit_boxing():
    with pytest.raises(IRSchemaError):
        concat_module(((8, 3), (8, 3)), axis=0, split=True)
