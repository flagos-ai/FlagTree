# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Friendly runtime wrappers must match the entry's structural ABI."""

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.runtime import module as runtime_module


@pytest.mark.parametrize("nested_input,nested_output", [(False, True), (True, False), (True, True)])
def test_aggregate_entry_selects_flat_buffer_call_graph_adapter(monkeypatch, nested_input, nested_output):
    value_type = fm.tensor_type("float32", (16,))
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    input_type = fm.TupleType((value_type,)) if nested_input else value_type
    source = builder.var("source", input_type)
    value = builder.call("builtin.get_item", (source,), value_type, attrs={"index": 0}) if nested_input else source
    result = builder.call("builtin.tuple", (value,), fm.TupleType((value_type,))) if nested_output else value
    builder.function("main", (source,), (result,))
    module = fm.verify_module(builder.build(entry="main"))
    marker = object()
    monkeypatch.setattr(runtime_module, "GeneratedTirCallGraphModule", lambda *args: marker)
    assert runtime_module.create_tir_runtime(None, None, module, None) is marker
