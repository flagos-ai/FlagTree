# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Reshape stays provable while a token extent is still symbolic."""

import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.passes.shape_bucket import bind_shape_buckets


def _reshape_module(tokens, requested, width=32):
    class Graph(fm.Module):
        def __init__(self):
            super().__init__(dialect='high_level', stage='imported', entry='main')

        def forward(self):
            value = self.input('value', fm.tensor_type('bfloat16', (tokens, width)), id='value')
            out = fm.F.tensors.reshape(value, requested, name='out')
            self.function('main', [value], [out])

    return Graph().build()


def test_symbolic_input_infers_a_symbolic_leading_dimension():
    module = _reshape_module(fm.dim('tokens', 1, 2), (-1, 4, 8))
    shape = fm.logical_type(module.node_map['out'].type).shape
    assert shape[0] == fm.dim('tokens', 1, 2)
    assert [dimension.fixed_value for dimension in shape[1:]] == [4, 8]


def test_bound_symbolic_reshape_evaluates_like_the_static_shape():
    from dataclasses import replace

    symbolic = _reshape_module(fm.dim('tokens', 1, 2), (-1, 4, 8))
    # The T=2 bucket must evaluate exactly like a static T=2 reshape.
    declared = replace(symbolic, metadata={
        'shape_bucket': {
            'range_info': {'tokens': [1, 2]},
            'segments_count': 2,
        },
    })
    bound = bind_shape_buckets(declared, segment=1)
    value = torch.arange(64, dtype=torch.float32).reshape(2, 32).bfloat16()
    result, = TorchEvaluator(DictWeightResolver({})).run(bound, {'value': value})
    assert result.shape == (2, 4, 8)
    assert torch.equal(result, value.reshape(2, 4, 8))


def test_symbolic_reshape_requires_a_provable_inferred_dimension():
    with pytest.raises(IRSchemaError, match='cannot prove the inferred dimension'):
        _reshape_module(fm.dim('tokens', 1, 2), (-1, 5, 7))


def test_static_reshape_error_is_unchanged():
    with pytest.raises(IRSchemaError, match='inferred dimension is not integral'):
        _reshape_module(2, (-1, 5, 7))
    with pytest.raises(IRSchemaError, match='changes logical element count from 64 to 140'):
        _reshape_module(2, (4, 5, 7))
