# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)
from triton.flagmega.ir import kernel_dispatch_of
from triton.flagmega.stages import get_stage, next_stage

from .helpers import StubTarget, semantic_packed_qkv_module


def _policy():
    return TritonMicroKernelSelectionPolicy(default_triton_microkernel_registry())


def test_policy_proposes_capability_filtered_agent_editable_point():
    proposed = _policy().propose(semantic_packed_qkv_module(), StubTarget())
    point = next(
        value for value in proposed.selection_points
        if value.id == "microkernel.packed_qkv"
    )

    assert point.kind == "tir_microkernel"
    assert point.owner is None
    assert point.default_candidate == "test.qkv.pipeline"
    assert proposed.selection_map[point.id].candidate_id == "test.qkv.pipeline"

    filtered = _policy().propose(
        semantic_packed_qkv_module(), StubTarget(supported=())
    )
    point = next(
        value for value in filtered.selection_points
        if value.id == "microkernel.packed_qkv"
    )
    assert tuple(value.id for value in point.candidates) == ("test.qkv.scalar",)


def test_agent_selection_materializes_microkernel_without_changing_semantic_tir():
    target = StubTarget()
    proposed = _policy().propose(semantic_packed_qkv_module(), target)
    edited = replace(
        proposed,
        selections=tuple(
            replace(record, candidate_id="test.qkv.scalar", origin="agent", policy="agent")
            if record.point_id == "microkernel.packed_qkv" else record
            for record in proposed.selections
        ),
    )
    selected = fm.verify_module(_policy().apply(edited, target))
    dispatch = kernel_dispatch_of(selected.prim_function_map["packed_qkv"])

    assert dispatch.semantic_op == "ntt.packed_qkv_parallel_linear_fused_rhs"
    assert dispatch.semantic_candidate == "ntt.packed_qkv_parallel_linear"
    assert dispatch.microkernel.implementation == "test.qkv.scalar"
    assert dispatch.microkernel.parameters == {"block_k": 64, "block_n": 8}


def test_selected_microkernel_survives_editable_python_checkpoint(tmp_path):
    target = StubTarget()
    selected = _policy().apply(
        _policy().propose(semantic_packed_qkv_module(), target), target
    )
    checkpoint = fm.emit_module(selected, tmp_path / "selected_qkv.py")
    resumed = fm.load_module(checkpoint)

    assert resumed.semantic_hash == selected.semantic_hash
    assert "T.microkernel_selection(" in checkpoint.read_text(encoding="utf-8")


def test_default_stage_order_places_microkernel_selection_after_canonicalization():
    target = StubTarget()
    module = replace(semantic_packed_qkv_module(), stage="selected_tir")

    canonical_stage = next_stage(module.stage)
    assert canonical_stage.name == "canonicalize-packed-qkv-weights"
    canonical = canonical_stage.run(module, target)
    proposed = get_stage("propose-microkernels").run(canonical, target)
    selected = get_stage("select-microkernels").run(proposed, target)
    dispatch = kernel_dispatch_of(selected.prim_function_map["packed_qkv"])

    assert canonical.stage == "canonicalized_tir"
    assert proposed.stage == "microkernel_candidates"
    assert selected.stage == "selected_microkernels"
    assert dispatch.microkernel.implementation == "test.qkv.pipeline"
