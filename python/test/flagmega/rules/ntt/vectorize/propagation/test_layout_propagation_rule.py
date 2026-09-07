# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules.ntt.vectorize.propagation import layout_propagation_rules


def _rule(name):
    return next(rule for rule in layout_propagation_rules() if rule.name == name)


def _assert_equivalent(original, rewritten, inputs):
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(original, inputs)[0],
    )


def test_vectorize_transpose_maps_lane_axis_through_permutation(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (2, 4, 16)), id="value"
            )
            permuted = fm.F.tensors.permute(value, (1, 0, 2), name="permuted")
            root = fm.F.tensors.pack(permuted, (8,), axes=(2,), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("VectorizeTransposePropagation").apply(
        module.node_map["root"], module
    )

    assert result is not None
    assert result.prefix_nodes[0].attrs["axes"] == (2,)
    assert result.replacement.op == "tensors.permute"
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(
        module,
        rewritten,
        {"value": torch.randn((2, 4, 16), dtype=torch.bfloat16)},
    )


def test_transpose_devectorize_maps_lane_axis_through_permutation(materialize_rewrite):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 4, 2)),
                id="value",
            )
            unpacked = fm.F.tensors.unpack(value, axes=(2,), name="unpacked")
            root = fm.F.tensors.permute(unpacked, (1, 0, 2), name="root")
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule("TransposeDevectorizePropagation").apply(
        module.node_map["root"], module
    )

    assert result is not None
    assert result.prefix_nodes[0].op == "tensors.permute"
    assert result.replacement.attrs["axes"] == (2,)
    rewritten = materialize_rewrite(module, result)
    _assert_equivalent(
        module,
        rewritten,
        {"value": torch.randn((2, 4, 2, 8), dtype=torch.bfloat16)},
    )


@pytest.mark.parametrize(
    "op,attrs,rule_name",
    [
        ("pad", {"pad_end": (0, 8)}, "VectorizePadPropagation"),
        ("slice", {"shape": (2, 8)}, "VectorizeSlicePropagation"),
    ],
)
def test_vectorize_shape_transform_scales_physical_attribute(
    materialize_rewrite, op, attrs, rule_name,
):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (2, 8 if op == "pad" else 16)),
                id="value",
            )
            transformed = (
                fm.F.tensors.pad(value, attrs["pad_end"], name="transformed")
                if op == "pad"
                else fm.F.tensors.slice_to_shape(
                    value, attrs["shape"], name="transformed"
                )
            )
            root = fm.F.tensors.pack(
                transformed, (8,), axes=(1,), name="root"
            )
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule(rule_name).apply(module.node_map["root"], module)

    assert result is not None
    attribute = "pad_end" if op == "pad" else "shape"
    expected = (0, 1) if op == "pad" else (2, 1)
    assert result.replacement.attrs[attribute] == expected
    rewritten = materialize_rewrite(module, result)
    shape = (2, 8 if op == "pad" else 16)
    _assert_equivalent(
        module,
        rewritten,
        {"value": torch.randn(shape, dtype=torch.bfloat16)},
    )


@pytest.mark.parametrize(
    "op,rule_name",
    [
        ("pad", "PadDevectorizePropagation"),
        ("slice", "SliceDevectorizePropagation"),
    ],
)
def test_shape_transform_devectorize_scales_physical_attribute(
    materialize_rewrite, op, rule_name,
):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            physical_width = 1 if op == "pad" else 2
            value = self.input(
                "value",
                fm.tensor_type(
                    fm.vector_type("bfloat16", (8,)), (2, physical_width)
                ),
                id="value",
            )
            unpacked = fm.F.tensors.unpack(value, axes=(1,), name="unpacked")
            root = (
                fm.F.tensors.pad(unpacked, (0, 8), name="root")
                if op == "pad"
                else fm.F.tensors.slice_to_shape(unpacked, (2, 8), name="root")
            )
            self.function("main", (value,), (root,))

    module = Graph().build()
    result = _rule(rule_name).apply(module.node_map["root"], module)

    assert result is not None
    attribute = "pad_end" if op == "pad" else "shape"
    expected = (0, 1) if op == "pad" else (2, 1)
    assert result.prefix_nodes[0].attrs[attribute] == expected
    assert result.replacement.op == "tensors.unpack"
    rewritten = materialize_rewrite(module, result)
    physical_width = 1 if op == "pad" else 2
    _assert_equivalent(
        module,
        rewritten,
        {
            "value": torch.randn(
                (2, physical_width, 8), dtype=torch.bfloat16
            )
        },
    )


@pytest.mark.parametrize(
    "op,rule_name",
    [
        ("pad", "VectorizePadPropagation"),
        ("slice", "VectorizeSlicePropagation"),
    ],
)
def test_vectorize_shape_transform_rejects_unaligned_source(op, rule_name):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value", fm.tensor_type("bfloat16", (2, 12)), id="value"
            )
            transformed = (
                fm.F.tensors.pad(value, (0, 4), name="transformed")
                if op == "pad"
                else fm.F.tensors.slice_to_shape(value, (2, 8), name="transformed")
            )
            root = fm.F.tensors.pack(
                transformed, (8,), axes=(1,), name="root"
            )
            self.function("main", (value,), (root,))

    module = Graph().build()

    assert _rule(rule_name).apply(module.node_map["root"], module) is None


@pytest.mark.parametrize(
    "op,rule_name",
    [
        ("pad", "PadDevectorizePropagation"),
        ("slice", "SliceDevectorizePropagation"),
    ],
)
def test_shape_transform_devectorize_rejects_unaligned_attribute(op, rule_name):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")

        def forward(self):
            value = self.input(
                "value",
                fm.tensor_type(fm.vector_type("bfloat16", (8,)), (2, 2)),
                id="value",
            )
            unpacked = fm.F.tensors.unpack(value, axes=(1,), name="unpacked")
            root = (
                fm.F.tensors.pad(unpacked, (0, 4), name="root")
                if op == "pad"
                else fm.F.tensors.slice_to_shape(unpacked, (2, 12), name="root")
            )
            self.function("main", (value,), (root,))

    module = Graph().build()

    assert _rule(rule_name).apply(module.node_map["root"], module) is None
