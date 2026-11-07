"""Test FlagTune autotuner dtype identity extraction."""

from __future__ import annotations

import torch
import pytest

from triton.flagtune.runtime import autotuner as autotuner_mod
from triton.flagtune.contract.identity import ModelIdentity
from triton.flagtune.runtime.autotuner import Flagtuner, _infer_tensor_dtypes


class _DtypeLookalike:
    """Expose a dtype attribute without representing a tensor argument."""

    dtype = torch.float16


def test_default_dtype_extraction_ignores_non_tensor_dtype_attributes():
    """Keep FlagTree's default identity aligned with FlagGems LibTuner."""
    tensor = torch.empty(1, dtype=torch.bfloat16)

    assert _infer_tensor_dtypes((tensor, _DtypeLookalike())) == (torch.bfloat16, )


def test_enabled_flagtune_propagates_model_initialization_failure(monkeypatch):
    """An explicitly enabled model contract must never degrade to Triton pruning."""
    identity = ModelIdentity("nvidia-h20", "flaggems/mm", "gemv", "bf16-bf16-bf16")
    tuner = Flagtuner.__new__(Flagtuner)
    tuner._flagtune_op_id = "flaggems/mm"
    tuner._flagtune_variant = "gemv"
    tuner._flagtune_models = {}
    monkeypatch.setattr("triton.flagtune.is_enabled", lambda: True)
    monkeypatch.setattr(
        "triton.flagtune.runtime.proposer.load_model_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad package")),
    )

    with pytest.raises(RuntimeError, match="bad package"):
        tuner._ensure_flagtune(identity)


def test_enabled_flagtune_propagates_identity_failure(monkeypatch):
    """Identity failures are contract failures when the named tuner is enabled."""
    tuner = Flagtuner.__new__(Flagtuner)
    tuner._flagtune_op_id = "flaggems/mm"
    tuner._flagtune_variant = "gemv"
    tuner.nargs = {}
    monkeypatch.setattr("triton.flagtune.is_enabled", lambda: True)
    monkeypatch.setattr(
        autotuner_mod.Autotuner,
        "prune_configs",
        lambda _self, _kwargs: ["baseline"],
    )
    monkeypatch.setattr(
        tuner,
        "_runtime_identity",
        lambda _kwargs: (_ for _ in ()).throw(ValueError("no identity")),
    )

    with pytest.raises(ValueError, match="no identity"):
        tuner.prune_configs({})
