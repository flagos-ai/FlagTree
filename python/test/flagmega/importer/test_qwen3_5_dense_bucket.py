# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The dense Qwen3.5 importer can declare a token shape bucket."""

import os

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.qwen3_5_dense import Qwen35DenseImporter, token_extent
from triton.flagmega.passes.shape_bucket import (
    bind_shape_buckets,
    module_dimension_symbols,
    shape_bucket_declaration,
)

GGUF = os.environ.get("FLAGMEGA_QWEN38_GGUF")


def test_token_extent_accepts_static_counts_and_ranges():
    assert token_extent(1) == (1, None)
    assert token_extent(2) == (2, None)
    extent, bounds = token_extent((1, 2))
    assert (extent.name, extent.minimum, extent.maximum) == ('tokens', 1, 2)
    assert bounds == (1, 2)
    extent, bounds = token_extent([2, 2])
    assert (extent.name, extent.minimum, extent.maximum) == ('tokens', 2, 2)
    assert bounds == (2, 2)


@pytest.mark.parametrize('value', [0, -1, True, (0, 2), (3, 2), (1,), (1, 2, 3), 'tokens', 2.0])
def test_token_extent_rejects_invalid_values(value):
    with pytest.raises(ImporterError, match='token'):
        token_extent(value)


@pytest.mark.skipif(not GGUF, reason="set FLAGMEGA_QWEN38_GGUF to the Qwen3.8-27B IQ4_XS GGUF")
def test_dense_bucket_import_declares_and_binds_the_token_range():
    from triton.flagmega.importer import GGUFCheckpoint

    checkpoint = GGUFCheckpoint(GGUF)
    symbolic = Qwen35DenseImporter(checkpoint, num_tokens=(1, 2)).import_module()
    assert symbolic.stage == 'imported'
    assert module_dimension_symbols(symbolic) == ('tokens',)
    declaration = shape_bucket_declaration(symbolic)
    assert declaration['range_info'] == {'tokens': (1, 2)}
    assert declaration['segments_count'] == 2
    assert symbolic.metadata['mode'] == 'decode-bucket'
    assert symbolic.metadata['tokens_per_call'] == 2
    assert fm.logical_type(symbolic.node_map['input_ids'].type).shape[0].name == 'tokens'
    # Every decoder signature carries the same symbolic extent.
    hidden = next(node for node in symbolic.nodes if node.id == 'decode_linear_0_hidden')
    assert hidden.type.shape[0].name == 'tokens'
    # Symbolic reshapes keep one inferred axis instead of a literal count.
    reshape = next(node for node in symbolic.nodes if node.op == 'tensors.reshape')
    assert reshape.attrs['shape'][0] == -1

    bound = bind_shape_buckets(symbolic, segment=1)
    assert module_dimension_symbols(bound) == ()
    assert fm.logical_type(bound.node_map['input_ids'].type).shape[0].fixed_value == 2
    assert bound.metadata['tokens_per_call'] == 2
    assert dict(bound.metadata['shape_bucket_binding']['bindings']) == {'tokens': 2}


@pytest.mark.skipif(not GGUF, reason="set FLAGMEGA_QWEN38_GGUF to the Qwen3.8-27B IQ4_XS GGUF")
def test_static_dense_import_has_no_bucket_declaration():
    from triton.flagmega.importer import GGUFCheckpoint

    checkpoint = GGUFCheckpoint(GGUF)
    module = Qwen35DenseImporter(checkpoint).import_module()
    assert shape_bucket_declaration(module) is None
    assert module.metadata['tokens_per_call'] == 1
    assert module.metadata['mode'] == 'decode-1'
    assert module_dimension_symbols(module) == ()
