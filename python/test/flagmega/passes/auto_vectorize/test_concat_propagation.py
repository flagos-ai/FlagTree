# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation.concat import concat_propagation_rules


def _run(module):
    return DataflowPass("ConcatPropagation", concat_propagation_rules()).run(module)


def test_pack_is_distributed_to_every_concat_input():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 8))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            joined = fm.F.tensors.concat(lhs, rhs, axis=1, name="joined")
            output = fm.F.tensors.pack(joined, (8,), axes=(1,), name="output")
            self.function("main", (lhs, rhs), (output,))

    original = Graph().build()
    result = _run(original)

    assert [node.op for node in result.nodes] == [
        "builtin.var",
        "builtin.var",
        "tensors.pack",
        "tensors.pack",
        "tensors.concat",
    ]
    lhs = torch.randn(2, 8, dtype=torch.bfloat16)
    rhs = torch.randn(2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"lhs": lhs, "rhs": rhs})[0],
        evaluator.run(original, {"lhs": lhs, "rhs": rhs})[0],
    )


def test_concat_moves_before_unpack_and_packs_scalar_siblings():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            vector_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 1))
            vector = self.input("vector", vector_type, id="vector")
            scalar = self.input("scalar", fm.tensor_type("bfloat16", (2, 8)), id="scalar")
            unpacked = fm.F.tensors.unpack(vector, axes=(1,), name="unpacked")
            output = fm.F.tensors.concat(unpacked, scalar, axis=1, name="output")
            self.function("main", (vector, scalar), (output,))

    original = Graph().build()
    result = _run(original)

    assert result.node_map["output"].op == "tensors.unpack"
    assert result.node_map["output.propagated.concat"].inputs == (
        "vector",
        "output.propagated.pack1",
    )
    vector = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    scalar = torch.randn(2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"vector": vector, "scalar": scalar})[0],
        evaluator.run(original, {"vector": vector, "scalar": scalar})[0],
    )
