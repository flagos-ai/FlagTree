# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Raw gate-log parameters survive typed emission and retain default hashes."""

from dataclasses import replace

import pytest
import torch
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.evaluator import DictWeightResolver, TorchEvaluator
from triton.flagmega.ir.ops.nn.kda_gated_recurrent_core import KdaGatedRecurrentCore
from triton.flagmega.ir.ops.nn.kda_gated_step import KdaGatedStep

from python.test.flagmega.codegen.triton.kernels.kda_gated_recurrent_core.test_kda_gated_recurrent_runtime import (
    _module,
    _values,
)


@pytest.mark.parametrize("phase", [None, False, True])
def test_log_rate_interpretation_and_resume(tmp_path, phase):
    module = _module(heads=2, dim=128, request_phase=phase is not None, decay_input="log_rate")
    restored = fm.load_module(fm.emit_module(module, tmp_path / "log_rate.py"))
    assert restored.semantic_hash == module.semantic_hash
    values = _values(torch, heads=2, dim=128)
    values["decay_rate"] = values["decay_rate"].log()
    if phase is not None:
        values["prefill"] = torch.tensor(phase)
    evaluator = TorchEvaluator(DictWeightResolver({}))
    actual = evaluator.run(restored, values)
    ordinary = fm.verify_module(
        replace(
            module,
            nodes=tuple(
                replace(n, attrs={k: v for k, v in n.attrs.items() if k != "decay_input"})
                if n.op in {"nn.kda_gated_step", "nn.kda_gated_recurrent_core"}
                else n
                for n in module.nodes
            ),
        )
    )
    expected = evaluator.run(ordinary, {**values, "decay_rate": values["decay_rate"].exp()})
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("request_phase", [False, True])
def test_default_rate_preserves_canonical_ir(request_phase):
    assert (
        _module(request_phase=request_phase, heads=2, dim=128).semantic_hash
        == _module(request_phase=request_phase, heads=2, dim=128, decay_input="rate").semantic_hash
    )


@pytest.mark.parametrize("definition", [KdaGatedRecurrentCore, KdaGatedStep])
@pytest.mark.parametrize("value", ["unknown", None, True, 1, [], {}])
def test_invalid_decay_interpretation_is_rejected(definition, value):
    with pytest.raises(IRSchemaError, match="decay_input"):
        definition.normalize_attrs({"lower_bound": -5.0, "decay_input": value})
