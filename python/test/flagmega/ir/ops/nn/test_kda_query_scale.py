# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Attention readout scaling follows normalization and leaves state intact."""

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda import KdaRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore

from python.test.flagmega.ir.ops.nn.test_kda_gated_recurrent_core import _inputs, _module


@pytest.mark.parametrize("l2norm", (False, True))
def test_query_scale_changes_the_readout_but_not_the_updated_state(l2norm):
    values = _inputs()
    evaluator = TorchEvaluator(DictWeightResolver({}))
    ordinary, state = evaluator.run(_module(l2norm=l2norm), values)
    scaled, scaled_state = evaluator.run(_module(l2norm=l2norm, query_scale=.5), values)
    # Scaling by a power of two commutes with FP32/BF16 rounding for these
    # normal-range values. This also catches scaling before L2 normalization.
    torch.testing.assert_close(scaled, ordinary * .5, rtol=0, atol=0)
    torch.testing.assert_close(scaled_state, state, rtol=0, atol=0)


@pytest.mark.parametrize("definition", (KdaRecurrentCore, KdaGatedRecurrentCore))
@pytest.mark.parametrize("scale", (False, 0, -1, float("inf"), float("nan"), "0.5"))
def test_query_scale_rejects_invalid_attention_scales(definition, scale):
    with pytest.raises(IRSchemaError, match="query_scale must be a positive finite"):
        definition.normalize_attrs({"lower_bound": -5.0, "query_scale": scale}
                                   if definition is KdaGatedRecurrentCore else {"query_scale": scale})


def test_query_scale_roundtrips_without_changing_legacy_default_ir(tmp_path):
    legacy = _module()
    assert legacy.semantic_hash == _module(query_scale=1.).semantic_hash
    assert all("query_scale" not in node.attrs for node in legacy.nodes
               if node.op == "nn.kda_gated_recurrent_core")
    scaled = _module(query_scale=.5)
    path = tmp_path / "scaled.py"
    fm.emit_module(scaled, path)
    restored = fm.load_module(path)
    assert scaled.semantic_hash == restored.semantic_hash
    actual = TorchEvaluator(DictWeightResolver({})).run(restored, _inputs())
    expected = TorchEvaluator(DictWeightResolver({})).run(scaled, _inputs())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
