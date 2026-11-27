# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes import DataflowPass
from triton.flagmega.rules.ntt.vectorize.propagation import propagation_rules


def test_pack_propagation_reaches_fixpoint_across_two_devectorized_operands():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            value_type = fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            lhs_scalar = fm.F.tensors.unpack(lhs, axes=(1,), name="lhs_scalar")
            rhs_scalar = fm.F.tensors.unpack(rhs, axes=(1,), name="rhs_scalar")
            root = fm.F.math.add(lhs_scalar, rhs_scalar, name="root")
            self.function("main", (lhs, rhs), (root,))

    module = Graph().build()
    result = DataflowPass("PackPropagation", propagation_rules(), max_iterations=32).run(module)
    assert [node.op for node in result.nodes] == [
        "builtin.var", "builtin.var", "math.vectorized_binary", "tensors.unpack",
    ]
    assert result.node_map["root.propagated.compute"].inputs == ("lhs", "rhs")
    assert result.node_map["root"].inputs == ("root.propagated.compute",)

    lhs = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    rhs = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(result, {"lhs": lhs, "rhs": rhs})[0],
        evaluator.run(module, {"lhs": lhs, "rhs": rhs})[0],
    )


def test_pack_propagation_folds_inverse_boundary_and_updates_function_output():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="vectorized", entry="main")

        def forward(self):
            source = self.input("source", fm.tensor_type("bfloat16", (2, 16)), id="source")
            packed = fm.F.tensors.pack(source, (8,), axes=(1,), name="packed")
            root = fm.F.tensors.unpack(packed, axes=(1,), name="root")
            self.function("main", (source,), (root,))

    result = DataflowPass("PackPropagation", propagation_rules()).run(Graph().build())
    assert [node.id for node in result.nodes] == ["source"]
    assert result.function_map["main"].outputs == ("source",)
