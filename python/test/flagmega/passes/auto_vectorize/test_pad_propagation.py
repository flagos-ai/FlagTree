# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.layout import layout_propagation_rules


def _run(module):
    return DataflowPass("PadPropagation", layout_propagation_rules()).run(module)


def test_pack_moves_before_lane_aligned_end_pad():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 8)), id="value")
            padded = fm.F.tensors.pad(value, (0, 8), name="padded")
            output = fm.F.tensors.pack(padded, (8,), axes=(1,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output"].op == "tensors.pad"
    assert result.node_map["output"].attrs["pad_end"] == (0, 1)
    value = torch.randn(2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_pad_moves_before_unpack_when_padding_is_lane_aligned():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 1))
            value = self.input("value", value_type, id="value")
            scalar = fm.F.tensors.unpack(value, axes=(1,), name="scalar")
            output = fm.F.tensors.pad(scalar, (0, 8), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output.propagated.pad"].attrs["pad_end"] == (0, 1)
    assert result.node_map["output"].op == "tensors.unpack"
    value = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_non_lane_aligned_pad_is_not_rewritten():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 15)), id="value")
            padded = fm.F.tensors.pad(value, (0, 1), name="padded")
            output = fm.F.tensors.pack(padded, (8,), axes=(1,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.semantic_hash == original.semantic_hash
