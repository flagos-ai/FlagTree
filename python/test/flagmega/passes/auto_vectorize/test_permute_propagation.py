# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.layout import layout_propagation_rules


def _run(module):
    return DataflowPass("PermutePropagation", layout_propagation_rules()).run(module)


def test_pack_moves_before_permute_with_inverse_input_axis():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 16)), id="value")
            transposed = fm.F.tensors.permute(value, (1, 0), name="transposed")
            output = fm.F.tensors.pack(transposed, (8,), axes=(0,), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert [node.op for node in result.nodes] == [
        "builtin.var",
        "tensors.pack",
        "tensors.permute",
    ]
    assert result.node_map["output.propagated.pack"].attrs["axes"] == (1,)
    value = torch.randn(2, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )


def test_permute_moves_before_unpack_and_remaps_output_axis():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2))
            value = self.input("value", value_type, id="value")
            scalar = fm.F.tensors.unpack(value, axes=(1,), name="scalar")
            output = fm.F.tensors.permute(scalar, (1, 0), name="output")
            self.function("main", (value,), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output"].op == "tensors.unpack"
    assert result.node_map["output"].attrs["axes"] == (0,)
    value = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"value": value})[0],
        evaluator.run(original, {"value": value})[0],
    )
