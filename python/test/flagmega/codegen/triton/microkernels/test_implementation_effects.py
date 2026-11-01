# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Late physical selection must update the effects consumed by placement."""

from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)

from .helpers import StubTarget, semantic_packed_qkv_module


@pytest.mark.parametrize("facts", (
    {"result_memory_scope": "chip"},
    {"source_memory_scope": "device"},
    {"participant_scope": "single_program"},
))
def test_semantic_microkernel_materializes_physical_effects(tmp_path, facts):
    target = StubTarget()
    model = target.triton_implementation_model
    target.triton_implementation_model = replace(model, implementations=tuple(
        replace(value, facts={**value.facts, **facts}) for value in model.implementations
    ))
    policy = TritonMicroKernelSelectionPolicy(default_triton_microkernel_registry())
    selected = fm.verify_module(policy.apply(policy.propose(semantic_packed_qkv_module(), target), target))
    restored = fm.load_module(fm.emit_module(selected, tmp_path / "selected.py"))
    assert restored.semantic_hash == selected.semantic_hash
    dispatch = fm.kernel_dispatch_of(restored.prim_function_map["packed_qkv"])
    for name, effect in dispatch.memory_effect_map.items():
        if "result_memory_scope" in facts and name in dispatch.outputs:
            assert effect.scope is fm.MemoryAccessScope.CHIP
        if "source_memory_scope" in facts and name in dispatch.arguments:
            assert effect.scope is fm.MemoryAccessScope.DEVICE
        if facts.get("participant_scope") == "single_program":
            assert effect.access_domain == fm.MemoryAccessDomain.fixed_block(0)
