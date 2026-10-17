# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.rules import DataflowRewriter
from triton.flagmega.rules.neutral import fold_pack_reshape_rule


def _run(module):
    return DataflowRewriter((fold_pack_reshape_rule(),)).rewrite(module)


def _assert_equivalent(original, rewritten, inputs):
    evaluator = TorchEvaluator(DictWeightResolver({}))
    torch.testing.assert_close(
        evaluator.run(rewritten, inputs)[0],
        evaluator.run(original, inputs)[0],
    )


def test_fold_pack_reshape_maps_split_to_source_axis():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1024,)), id="value")
            reshaped = fm.F.tensors.reshape(value, (8, 128), name="reshaped")
            result = fm.F.tensors.pack(reshaped, (8,), axes=(1,), name="result")
            self.function("main", (value,), (result,))

    original = Graph().build()
    rewritten = _run(original)
    packed = rewritten.node_map["result.fold_pack_reshape.pack"]
    assert packed.inputs == ("value",)
    assert packed.attrs["axes"] == (0,)
    assert rewritten.node_map["result"].attrs["shape"] == (8, 16)
    _assert_equivalent(
        original,
        rewritten,
        {"value": torch.randn(1024, dtype=torch.bfloat16)},
    )


def test_fold_pack_reshape_maps_across_trailing_unit_axis_like_nncase():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (1, 16, 128)), id="value")
            reshaped = fm.F.tensors.reshape(value, (16, 128, 1), name="reshaped")
            result = fm.F.tensors.pack(reshaped, (8,), axes=(1,), name="result")
            self.function("main", (value,), (result,))

    original = Graph().build()
    rewritten = _run(original)
    packed = rewritten.node_map["result.fold_pack_reshape.pack"]
    assert packed.attrs["axes"] == (2,)
    assert rewritten.node_map["result"].attrs["shape"] == (16, 16, 1)
    _assert_equivalent(
        original,
        rewritten,
        {"value": torch.randn(1, 16, 128, dtype=torch.bfloat16)},
    )


def test_fold_pack_reshape_rejects_vector_lane_in_middle_of_mapped_shape():
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect="ntt", stage="packed", entry="main")

        def forward(self):
            value = self.input("value", fm.tensor_type("bfloat16", (4, 8)), id="value")
            reshaped = fm.F.tensors.reshape(value, (8, 4), name="reshaped")
            result = fm.F.tensors.pack(reshaped, (2,), axes=(0,), name="result")
            self.function("main", (value,), (result,))

    original = Graph().build()
    assert _run(original).semantic_hash == original.semantic_hash


@pytest.mark.parametrize('source_shape,target_shape', [((128,), (2, 64)), ((2, 8, 8), (2, 64))])
def test_fold_pack_reshape_composes_repeated_axis_lanes(source_shape, target_shape):
    class Graph(fm.Module):
        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', source_shape), id='value')
            reshaped = fm.F.tensors.reshape(value, target_shape, name='reshaped')
            packed = fm.F.tensors.pack(reshaped, (2, 4), axes=(1, 1), name='result')
            self.function('main', (value,), (packed,))

    original = Graph(dialect='ntt', stage='packed', entry='main').build()
    rewritten = _run(original)
    assert rewritten.node_map['result'].op == 'tensors.reshape'
    assert rewritten.node_map['result'].attrs['shape'] == (2, 8)
    value = torch.arange(128, dtype=torch.bfloat16).reshape(source_shape)
    _assert_equivalent(original, rewritten, {'value': value})
