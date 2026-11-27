# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.reshape import reshape_propagation_rules


def _run(module):
    return DataflowPass("ReshapePropagation", reshape_propagation_rules()).run(module)


def test_pack_moves_before_row_major_split_reshape():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1024,)), id="value")
            reshaped = fm.F.tensors.reshape(value, (8, 128), name="reshaped")
            output = fm.F.tensors.pack(reshaped, (8,), axes=(1,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output.propagated.pack"].attrs["axes"] == (0,)
    assert result.node_map["output"].op == "tensors.reshape"
    assert result.node_map["output"].attrs["shape"] == (8, 16)
    value = torch.randn(1024, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_reshape_moves_before_unpack_when_scalar_stride_maps():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (8, 16))
            value = self.input("value", value_type, id="value")
            scalar = fm.F.tensors.unpack(value, axes=(1,), name="scalar")
            output = fm.F.tensors.reshape(scalar, (1024,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output.propagated.reshape"].attrs["shape"] == (128,)
    assert result.node_map["output"].op == "tensors.unpack"
    assert result.node_map["output"].attrs["axes"] == (0,)
    value = torch.randn(8, 16, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_middle_stride_vector_axis_is_conservatively_rejected():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1024,)), id="value")
            reshaped = fm.F.tensors.reshape(value, (8, 128), name="reshaped")
            output = fm.F.tensors.pack(reshaped, (8,), axes=(0,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    assert _run(original).semantic_hash == original.semantic_hash
