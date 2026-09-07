# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator


def test_pack_unpack_boundaries_are_eliminated_between_vectorized_ops():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            lhs = self.input("lhs", fm.tensor_type("bfloat16", (1, 16)), id="lhs")
            rhs = self.input("rhs", fm.tensor_type("bfloat16", (1, 16)), id="rhs")
            added = fm.F.math.add(lhs, rhs, name="added")
            output = fm.F.math.silu(added, name="output")
            self.function("main", (lhs, rhs), (output,))

    module = Graph().build()
    vectorized = Compiler().compile(module, stop_after="apply-vectorization").module
    assert sum(node.op == "tensors.pack" for node in vectorized.nodes) == 2
    assert sum(node.op == "tensors.unpack" for node in vectorized.nodes) == 1
    assert {node.op for node in vectorized.nodes}.issuperset({"math.vectorized_binary", "math.vectorized_unary"})

    lhs = torch.randn(1, 16, dtype=torch.bfloat16)
    rhs = torch.randn(1, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(vectorized, {"lhs": lhs, "rhs": rhs})[0],
        torch.nn.functional.silu(lhs + rhs),
    )

    lowered = Compiler().compile(module).module
    calls = [
        node
        for node in lowered.nodes
        if fm.kernel_dispatch_for_call(lowered, node) is not None
    ]
    kernels = [fm.kernel_dispatch_for_call(lowered, node) for node in calls]
    compute_pairs = [
        (call, kernel)
        for call, kernel in zip(calls, kernels)
        if kernel.semantic_op in {
            "math.vectorized_binary", "math.vectorized_unary"
        }
    ]
    calls = [call for call, _ in compute_pairs]
    kernels = [kernel for _, kernel in compute_pairs]
    assert [node.semantic_op for node in kernels] == [
        "math.vectorized_binary", "math.vectorized_unary"
    ]
    # AutoDistribution sees the typed-vector producer/consumer relation and
    # selects one common layout.  Explicit contract lowering therefore keeps
    # a direct data edge; it must not synthesize a pack/unpack or boxing bridge.
    assert calls[1].inputs == (calls[0].id,)
    assert calls[1].type == calls[0].type
