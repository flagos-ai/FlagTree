# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize.propagation import unary_propagation_rules


def _rule(name):
    return next(rule for rule in unary_propagation_rules() if rule.name == name)


def test_vectorize_unary_propagation_pushes_pack_to_operand(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 16)), id="value")
            scalar = fm.F.math.silu(value, name="scalar")
            root = fm.F.tensors.pack(scalar, (8,), axes=(1,), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("VectorizeUnaryPropagation").apply(module.node_map["root"], module)
    assert result is not None
    assert [node.op for node in result.prefix_nodes] == ["tensors.pack"]
    assert result.replacement.op == "math.vectorized_unary"
    rewritten = materialize_rewrite(module, result)

    value = torch.randn(2, 16, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"value": value})[0],
        evaluator.run(module, {"value": value})[0],
    )


def test_unary_devectorize_propagation_moves_unpack_after_unary(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            vector = self.input(
                "vector", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)), id="vector",
            )
            unpacked = fm.F.tensors.unpack(vector, axes=(1,), name="unpacked")
            root = fm.F.math.silu(unpacked, name="root")
            self.function("main", (vector,), (root,))

    module = Graph().build()
    result = _rule("UnaryDevectorizePropagation").apply(module.node_map["root"], module)
    assert result is not None
    assert [node.op for node in result.prefix_nodes] == ["math.vectorized_unary"]
    assert result.replacement.op == "tensors.unpack"
    rewritten = materialize_rewrite(module, result)

    vector = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, {"vector": vector})[0],
        evaluator.run(module, {"vector": vector})[0],
    )


def test_unary_rules_reject_wrong_boundary_direction():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (16,)), id="value")
            root = fm.F.math.silu(value, name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    assert _rule("VectorizeUnaryPropagation").apply(module.node_map["root"], module) is None
    assert _rule("UnaryDevectorizePropagation").apply(module.node_map["root"], module) is None
