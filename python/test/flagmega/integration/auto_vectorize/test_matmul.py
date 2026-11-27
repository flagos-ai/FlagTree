# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def test_matmul_n_candidate_pads_and_vectorizes_output_axis():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (5, 7)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (7, 10)), id="rhs")
            output = fm.F.math.matmul(lhs, rhs, name="output")
            self.function("main", (lhs, rhs), (output,))

    module = Graph().build()
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    assert vectorized.selection_map["vectorization.output"].candidate_id == "vectorization.matmul.n"
    assert any(node.op == "math.vectorized_matmul" for node in vectorized.nodes)
    assert vectorized.node_map["output"].op == "tensors.slice_to_shape"

    lhs = torch.randn(5, 7, dtype=torch.bfloat16)
    rhs = torch.randn(7, 10, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"lhs": lhs, "rhs": rhs})[0],
        evaluator.run(module, {"lhs": lhs, "rhs": rhs})[0],
    )
