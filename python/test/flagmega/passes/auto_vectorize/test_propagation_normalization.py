# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.auto_distributed.policy import lower_vectorization_contracts
from triton.flagmega.passes.auto_vectorize import AutoVectorizePass
from triton.flagmega.targets import NvidiaSm90Target


def _round_trip(module, inputs):
    target = NvidiaSm90Target()
    vectorized = AutoVectorizePass.run(AutoVectorizePass.propose(module, target), target)
    normalized = fm.verify_module(lower_vectorization_contracts(vectorized))
    contracted = tuple(
        node for node in normalized.nodes
        if "selected_vectorization" in node.metadata
    )
    assert contracted
    assert all(node.metadata["selected_vector_axes"] for node in contracted)
    assert all(node.metadata["selected_vector_lanes"] for node in contracted)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(normalized, inputs)[0],
        evaluator.run(module, inputs)[0],
    )
    return vectorized, normalized


def test_permute_propagation_normalizes_to_original_semantic_graph():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 16)), id="value")
            activated = fm.F.math.silu(value, name="activated")
            output = fm.F.tensors.permute(activated, (1, 0), name="output")
            self.function("main", (value,), (output,))

    vectorized, normalized = _round_trip(
        Graph().build(), {"value": torch.randn(2, 16, dtype=torch.bfloat16)}
    )
    assert any(node.metadata.get("vectorization_rule") == "TransposeDevectorizePropagation" for node in vectorized.nodes)
    assert [node.op for node in normalized.nodes] == [
        "builtin.var",
        "tensors.pack",
        "math.vectorized_unary",
        "tensors.unpack",
        "tensors.permute",
    ]


def test_pad_and_slice_propagation_normalize_to_original_semantic_graph():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (2, 16)), id="value")
            activated = fm.F.math.silu(value, name="activated")
            padded = fm.F.tensors.pad(activated, (0, 8), name="padded")
            output = fm.F.tensors.slice_to_shape(padded, (2, 16), name="output")
            self.function("main", (value,), (output,))

    _, normalized = _round_trip(
        Graph().build(), {"value": torch.randn(2, 16, dtype=torch.bfloat16)}
    )
    assert [node.op for node in normalized.nodes] == [
        "builtin.var",
        "tensors.pack",
        "math.vectorized_unary",
        "tensors.unpack",
        "tensors.pad",
        "tensors.slice_to_shape",
    ]


def test_reshape_propagation_normalizes_to_original_semantic_graph():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (8, 128)), id="value")
            activated = fm.F.math.silu(value, name="activated")
            output = fm.F.tensors.reshape(activated, (1024,), name="output")
            self.function("main", (value,), (output,))

    _, normalized = _round_trip(
        Graph().build(), {"value": torch.randn(8, 128, dtype=torch.bfloat16)}
    )
    assert [node.op for node in normalized.nodes] == [
        "builtin.var",
        "tensors.pack",
        "math.vectorized_unary",
        "tensors.unpack",
        "tensors.reshape",
    ]


def test_concat_propagation_normalizes_both_vectorized_inputs():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="decomposed", entry="main")

        def forward(self):
            value_type = fm.tensor_type("bfloat16", (2, 16))
            lhs = self.input("lhs", value_type, id="lhs")
            rhs = self.input("rhs", value_type, id="rhs")
            lhs_value = fm.F.math.silu(lhs, name="lhs_value")
            rhs_value = fm.F.math.silu(rhs, name="rhs_value")
            output = fm.F.tensors.concat(lhs_value, rhs_value, axis=1, name="output")
            self.function("main", (lhs, rhs), (output,))

    _, normalized = _round_trip(
        Graph().build(),
        {
            "lhs": torch.randn(2, 16, dtype=torch.bfloat16),
            "rhs": torch.randn(2, 16, dtype=torch.bfloat16),
        },
    )
    assert [node.op for node in normalized.nodes] == [
        "builtin.var",
        "builtin.var",
        "tensors.pack",
        "math.vectorized_unary",
        "tensors.pack",
        "math.vectorized_unary",
        "tensors.unpack",
        "tensors.unpack",
        "tensors.concat",
    ]
