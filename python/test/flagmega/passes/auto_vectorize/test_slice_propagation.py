# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.layout import layout_propagation_rules


def _run(module):
    return DataflowPass("SlicePropagation", layout_propagation_rules()).run(module)


def test_pack_moves_before_lane_aligned_slice_to_shape():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 24)), id="value")
            sliced = fm.F.tensors.slice_to_shape(value, (2, 16), name="sliced")
            output = fm.F.tensors.pack(sliced, (8,), axes=(1,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output"].op == "tensors.slice_to_shape"
    assert result.node_map["output"].attrs["shape"] == (2, 2)
    value = torch.randn(2, 24, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_slice_moves_before_unpack_when_shape_is_lane_aligned():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 3))
            value = self.input("value", value_type, id="value")
            scalar = fm.F.tensors.unpack(value, axes=(1,), name="scalar")
            output = fm.F.tensors.slice_to_shape(scalar, (2, 16), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output.propagated.slice_to_shape"].attrs["shape"] == (2, 2)
    assert result.node_map["output"].op == "tensors.unpack"
    value = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
