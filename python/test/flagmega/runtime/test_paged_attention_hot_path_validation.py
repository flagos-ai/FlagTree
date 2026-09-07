# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import MethodType, SimpleNamespace

import pytest

import triton.flagmega.runtime.module as runtime_module
from triton.flagmega.runtime.module import GeneratedTirPagedAttentionLayerModule
from triton.flagmega.runtime.lifecycle import RuntimeModule


def _runtime(validation_calls):
    runtime = object.__new__(GeneratedTirPagedAttentionLayerModule)
    RuntimeModule.__init__(runtime)
    runtime._mark_loaded("cpu")
    runtime.ir_module = object()

    def validate(self, input_ids, state, *, dynamic_values=True):
        validation_calls.append(dynamic_values)

    runtime._validate_inputs = MethodType(validate, runtime)
    runtime._entry_buffer_values = MethodType(
        lambda self, input_ids, state, outputs: {}, runtime
    )
    runtime._prepare_bound = MethodType(lambda self, values: None, runtime)
    runtime._launch_bound = MethodType(
        lambda self, values, *, stream=None: None, runtime
    )
    return runtime


def test_prepare_checks_dynamic_state_but_run_only_checks_structural_contract(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    calls = []
    runtime = _runtime(calls)
    output = torch.empty((1,), dtype=torch.float32)
    input_ids = torch.empty((1,), dtype=torch.int32)
    state = SimpleNamespace()
    monkeypatch.setattr(
        runtime_module,
        "_single_tensor_output",
        lambda module: ("output", object()),
    )
    monkeypatch.setattr(runtime_module, "_validate_ir_tensor", lambda *args: None)

    runtime.prepare(input_ids, state, output=output)
    runtime.run_into(output, input_ids, state)

    assert calls == [True, False]
