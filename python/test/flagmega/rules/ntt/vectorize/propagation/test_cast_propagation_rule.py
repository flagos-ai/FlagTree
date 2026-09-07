# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize.propagation import cast_propagation_rules


def _rule(name):
    return next(rule for rule in cast_propagation_rules() if rule.name == name)


def _assert_equivalent(original, rewritten, inputs):
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(original, inputs)[0],
    )


def test_vectorize_cast_scales_input_lanes_by_element_width(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (32,)), id="value")
            cast = fm.F.tensors.cast(value, fm.DType.FLOAT32, name="cast")
            root = fm.F.tensors.pack(cast, (8,), axes=(0,), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("VectorizeCastPropagation").apply(module.node_map["root"], module)
    assert result is not None
    assert result.prefix_nodes[0].attrs["lanes"] == (16,)
    assert result.replacement.op == "ntt.vectorized_cast"
    assert result.replacement.type == fm.tensor_type(fm.vector_type("float32", (8,)), (4,))
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(module, rewritten, {"value": torch.randn(32, dtype=torch.bfloat16)})


def test_cast_devectorize_scales_output_lanes_by_element_width(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type(fm.vector_type("bfloat16", (16,)), (2,)), id="value"
            )
            unpacked = fm.F.tensors.unpack(value, axes=(0,), name="unpacked")
            root = fm.F.tensors.cast(unpacked, fm.DType.FLOAT32, name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("CastDevectorizePropagation").apply(module.node_map["root"], module)
    assert result is not None
    vectorized = result.prefix_nodes[0]
    assert vectorized.op == "ntt.vectorized_cast"
    assert vectorized.type == fm.tensor_type(fm.vector_type("float32", (8,)), (4,))
    assert result.replacement.op == "tensors.unpack"
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(
        module,
        rewritten,
        {"value": torch.randn(2, 16, dtype=torch.bfloat16)},
    )


def test_vectorize_cast_rejects_fractional_lane_scaling():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("int64", (8,)), id="value")
            cast = fm.F.tensors.cast(value, fm.DType.BFLOAT16, name="cast")
            root = fm.F.tensors.pack(cast, (1,), axes=(0,), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    assert _rule("VectorizeCastPropagation").apply(module.node_map["root"], module) is None


def test_fold_nop_vectorized_cast_returns_input_definition():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2,)), id="value"
            )
            root = fm.F.ntt.vectorized_cast(
                value, fm.vector_type("bfloat16", (8,)), (0,), name="root"
            )
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("FoldNopVectorizedCast").apply(module.node_map["root"], module)
    assert result is not None
    assert result.replacement.type == module.node_map["value"].type


def test_pack_cast_supports_nested_lane_groups_on_one_axis(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("float32", (64,)), id="value")
            cast = fm.F.tensors.cast(value, fm.DType.BFLOAT16, name="cast")
            root = fm.F.tensors.pack(cast, (4, 8), axes=(0, 0), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("VectorizeCastPropagation").apply(module.node_map["root"], module)
    assert result is not None
    assert result.replacement.type == module.node_map["root"].type
    _assert_equivalent(module, materialize_rewrite(module, result), {"value": torch.randn(64)})
