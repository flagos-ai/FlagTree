# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Identity-safe verification caching for immutable editable IR."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import verify as verify_impl


def _module(*, stage: str = "imported") -> fm.IRModule:
    value = fm.Node(
        "value",
        "builtin.var",
        (),
        fm.tensor_type("bfloat16", (1, 8)),
        attrs={"name": "value"},
    )
    return fm.IRModule(
        dialect="high_level",
        stage=stage,
        nodes=(value,),
        functions=(fm.Function("main", (value.id,), (value.id,)),),
        entry="main",
    )


def test_same_immutable_module_is_structurally_verified_once(monkeypatch):
    module = _module()
    calls = 0
    original = verify_impl._verify_type

    def count(value_type, stage, node_id):
        nonlocal calls
        calls += 1
        return original(value_type, stage, node_id)

    monkeypatch.setattr(verify_impl, "_verify_type", count)

    assert verify_impl.verify_module(module) is module
    assert verify_impl.verify_module(module) is module
    assert calls == 1


def test_equal_reconstructed_module_is_verified_independently(monkeypatch):
    first = _module()
    second = _module()
    calls = 0
    original = verify_impl._verify_type

    def count(value_type, stage, node_id):
        nonlocal calls
        calls += 1
        return original(value_type, stage, node_id)

    monkeypatch.setattr(verify_impl, "_verify_type", count)

    verify_impl.verify_module(first)
    verify_impl.verify_module(second)
    assert calls == 2


def test_cached_module_still_checks_requested_stage_and_dialect():
    module = _module()
    verify_impl.verify_module(module)

    with pytest.raises(IRVerificationError, match="Expected stage"):
        verify_impl.verify_module(module, expected_stage="distributed")
    with pytest.raises(IRVerificationError, match="Expected dialect"):
        verify_impl.verify_module(module, expected_dialect="ntt")
