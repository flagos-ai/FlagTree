# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.errors import IRVerificationError
from python.test.flagmega.passes.tir.bufferize.test_ref_slice import state_slice_graph


@pytest.fixture(scope="module")
def compiled():
    return Compiler().compile(state_slice_graph()).module


@pytest.mark.parametrize("edit", ["unbound", "missing_buffer", "tensor_buffer", "unused_symbol", "foreign_function"])
def test_verifier_rejects_invalid_executable_offset_bindings(compiled, edit):
    plan = fm.verify_buffer_plan(compiled)
    buffer = plan.buffer_map["view.convolution"]
    symbol, scalar = buffer.offset_bindings[0]
    changes = {
        "unbound": {"offset_bindings": ()},
        "missing_buffer": {"offset_bindings": ((symbol, "absent"), )},
        "tensor_buffer": {"offset_bindings": ((symbol, "qkv"), )},
        "unused_symbol": {"offset_bindings": ((symbol, scalar), ("unused", scalar))},
        "foreign_function": {"function": "different_function"},
    }
    edited = replace(buffer, **changes[edit])
    edited_plan = replace(plan, buffers=tuple(edited if value.id == buffer.id else value for value in plan.buffers))
    module = replace(compiled, metadata={**compiled.metadata, "buffer_plan": edited_plan.to_data()})
    with pytest.raises(IRVerificationError, match="offset binding"):
        fm.verify_buffer_plan(module)


def test_executable_memspan_bindings_roundtrip_through_editable_python_ir(compiled, tmp_path):
    path = tmp_path / "module.py"
    fm.emit_module(compiled, path)
    loaded = fm.load_module(path)
    assert loaded.semantic_hash == compiled.semantic_hash
    assert fm.verify_buffer_plan(loaded) == fm.verify_buffer_plan(compiled)
