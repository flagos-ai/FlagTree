# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Single-token chunk arithmetic has observable BF16 numerical boundaries."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
    _values,
)


def test_chunk_boundary_is_observable_and_exports_bf16_state(tmp_path):
    values = _values(torch, heads=2, dim=128)
    values['state'].zero_()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    module = _module(heads=2, dim=128, arithmetic='chunk_bf16', query_scale=128 ** -.5)
    path = tmp_path / 'chunk.py'
    fm.emit_module(module, path)
    restored = fm.load_module(path)
    assert restored.semantic_hash == module.semantic_hash
    output, state = evaluator.run(restored, values)
    recurrent, recurrent_state = evaluator.run(_module(heads=2, dim=128, query_scale=128 ** -.5), values)
    assert torch.count_nonzero(output != recurrent) > 0
    assert torch.count_nonzero(state != recurrent_state) > 0
    # FlashKDA's final state export rounds to BF16 even though the cache is FP32.
    assert torch.equal(state, state.bfloat16().float())


def test_default_arithmetic_keeps_existing_hash_and_omits_attribute():
    module = _module()
    assert module.semantic_hash == _module(arithmetic='recurrent').semantic_hash
    assert all('arithmetic' not in node.attrs for node in module.nodes)


@pytest.mark.parametrize('arithmetic', ['unknown', False, 1, None])
def test_unknown_arithmetic_is_rejected(arithmetic):
    with pytest.raises(IRSchemaError, match='arithmetic'):
        _module(arithmetic=arithmetic)


def test_chunk_rejects_unsupported_dimensions():
    with pytest.raises(IRSchemaError, match='128'):
        _module(arithmetic='chunk_bf16', dim=4)
