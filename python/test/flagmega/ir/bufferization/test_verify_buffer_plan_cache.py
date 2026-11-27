# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from importlib import import_module

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError


def _bufferized_module():
    builder = fm.IRBuilder(dialect="tir", stage="selected_tir")
    value_type = fm.tensor_type("bfloat16", (1, 128))
    source = builder.var("source", value_type, id="source")
    output = builder.call("test.unary", (source,), value_type, id="output")
    builder.function("main", (source,), (output,))
    module = builder.build(entry="main")
    plan = fm.make_buffer_plan(module)
    return replace(
        module,
        dialect="bufferized_tir",
        stage="bufferized_tir",
        metadata={**dict(module.metadata), "buffer_plan": plan.to_data()},
    )


def test_buffer_plan_parsing_is_shared_but_validation_is_module_specific(monkeypatch):
    verifier = import_module("triton.flagmega.ir.bufferization.verify")
    plan_module = import_module("triton.flagmega.ir.bufferization.plan")
    original = plan_module.BufferPlan.from_data
    calls = 0

    def counted(data):
        nonlocal calls
        calls += 1
        return original(data)

    module = _bufferized_module()
    reconstructed = replace(module)
    verifier._VERIFIED_BUFFER_PLANS.clear()
    verifier._PARSED_BUFFER_PLANS.clear()
    monkeypatch.setattr(plan_module.BufferPlan, "from_data", staticmethod(counted))
    try:
        first = fm.verify_buffer_plan(module)
        assert fm.verify_buffer_plan(module) is first
        assert fm.verify_buffer_plan(reconstructed) == first
        assert calls == 1

        incompatible = replace(module, functions=())
        with pytest.raises(IRVerificationError, match="exactly one ABI"):
            fm.verify_buffer_plan(incompatible)
        assert calls == 1
    finally:
        verifier._VERIFIED_BUFFER_PLANS.clear()
        verifier._PARSED_BUFFER_PLANS.clear()
