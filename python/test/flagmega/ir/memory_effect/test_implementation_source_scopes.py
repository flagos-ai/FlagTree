# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Input-specific physical scopes must reach both selected-TIR paths."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.memory_effect import implementation_memory_effect
from triton.flagmega.passes.tir import materialize_kernel_prim_functions

from python.test.flagmega.codegen.triton.microkernels.helpers import StubTarget, semantic_packed_qkv_module
from python.test.flagmega.passes.tir.test_materialize_kernel_functions import _two_add_kernels


def selected_module(path, scopes):
    if path == "semantic":
        target = StubTarget()
        model = target.triton_implementation_model
        target.triton_implementation_model = replace(model, implementations=tuple(
            replace(value, facts={**value.facts, "source_memory_scope": "chip", "source_memory_scopes": scopes})
            for value in model.implementations))
        policy = TritonMicroKernelSelectionPolicy(default_triton_microkernel_registry())
        return fm.verify_module(policy.apply(policy.propose(semantic_packed_qkv_module(), target), target))
    module = _two_add_kernels()
    module = replace(module, nodes=tuple(
        replace(node, attrs={**node.attrs, "facts": {**node.attrs["facts"], "source_memory_scope": "chip",
            "source_memory_scopes": scopes}}) if node.op == "tir.kernel" else node for node in module.nodes))
    return fm.verify_module(materialize_kernel_prim_functions(module))


@pytest.mark.parametrize("path,local,peer", [("semantic", "input", "fused_weight"), ("legacy", "lhs", "rhs")])
@pytest.mark.parametrize("scope", ["block", "inferred"])
def test_named_source_scope_keeps_other_inputs_shared(tmp_path, path, local, peer, scope):
    selected = selected_module(path, {local: scope})
    restored = fm.load_module(fm.emit_module(selected, tmp_path / "selected.py"))
    assert restored.semantic_hash == selected.semantic_hash
    call = next(node for node in restored.nodes if node.op == "tir.call")
    dispatch = fm.kernel_dispatch_for_call(restored, call)
    assert dispatch.memory_effect_map[local].scope.value == scope
    assert dispatch.memory_effect_map[peer].scope is fm.MemoryAccessScope.CHIP
    assert all(dispatch.memory_effect_map[name].scope is fm.MemoryAccessScope.INFERRED for name in dispatch.outputs)


@pytest.mark.parametrize("path", ["semantic", "legacy"])
@pytest.mark.parametrize("bad", [{"misspelled": "chip"}, {"result": "chip"}, "block", {"input": "rack"}, None])
def test_named_source_scope_rejects_missing_schema_or_invalid_scope(path, bad):
    with pytest.raises(IRVerificationError, match="source_memory_scopes"):
        selected_module(path, bad)


def test_named_inferred_scope_retains_the_op_requirement_and_none_stays_none():
    facts = {"source_memory_scope": "device", "source_memory_scopes": {"value": "inferred"}}
    kwargs = {"parameter_name": "value", "input_names": ("value",)}
    assert implementation_memory_effect(fm.MemoryEffect.CHIP_READ, facts, **kwargs) == fm.MemoryEffect.CHIP_READ
    assert implementation_memory_effect(fm.MemoryEffect.NONE, facts, **kwargs) == fm.MemoryEffect.NONE


@pytest.mark.parametrize("kwargs", [{}, {"input_names": ("value",)}])
def test_named_scopes_cannot_silently_omit_the_input_schema_or_name(kwargs):
    with pytest.raises(IRVerificationError, match="source_memory_scopes"):
        implementation_memory_effect(fm.MemoryEffect.READ, {"source_memory_scopes": {"value": "block"}}, **kwargs)
