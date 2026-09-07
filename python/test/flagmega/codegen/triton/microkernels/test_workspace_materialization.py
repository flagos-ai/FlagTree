# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import kernel_dispatch_of

from .helpers import StubTarget, semantic_packed_qkv_module


def _implementation(*, source_index=1):
    workspaces = (
        fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("bfloat16", (2, 64, 16)), 16
        ),
        fm.T.shared_workspace_descriptor(
            "accumulator", fm.tensor_type("float32", (64,)), 16
        ),
    )
    pipeline = fm.T.transfer_pipeline_contract(
        (fm.T.transfer_pipeline_channel(
            "rhs",
            source_argument_indices=(source_index,),
            shared_workspace_indices=(0,),
            source_alignment_bytes=16,
        ),),
        consumer_shared_workspace_indices=(1,),
    )
    return TritonImplementation(
        "test.qkv.workspace_pipeline",
        "qkv_parallel_linear",
        "workspace_pipeline",
        {"block_n": 64, "block_k": 128, "stages": 2},
        {"input_kind": "fused_rhs", "rhs_layout": "k_major"},
        requires=("async_matrix",),
        shared_workspaces=workspaces,
        transfer_pipeline=pipeline,
    )


def _target(*, source_index=1):
    target = StubTarget()
    implementation = _implementation(source_index=source_index)
    target.triton_implementation_model = TritonImplementationModel(
        (implementation,),
        {"qkv_parallel_linear": (implementation.id,)},
        "test-workspace-machine/v1",
    )
    return target


def _policy():
    return TritonMicroKernelSelectionPolicy(default_triton_microkernel_registry())


def test_selection_materializes_shared_buffers_and_transfer_contract(tmp_path):
    target = _target()
    selected = fm.verify_module(_policy().apply(
        _policy().propose(semantic_packed_qkv_module(), target), target
    ))
    dispatch = kernel_dispatch_of(selected.prim_function_map["packed_qkv"])

    assert tuple(value.name for value in dispatch.microkernel.shared_workspaces) == (
        "rhs_stage", "accumulator"
    )
    assert dispatch.microkernel.transfer_pipeline.channels[0].name == "rhs"
    assert tuple(value.name for value in dispatch.shared_workspace_buffers) == (
        "rhs_stage", "accumulator"
    )
    assert all(
        value.mem_span.buffer.memory_space == "shared"
        for value in dispatch.shared_workspace_buffers
    )
    assert dispatch.shared_workspace_buffers[0].mem_span.size.fixed_value == 4096

    checkpoint = fm.emit_module(selected, tmp_path / "selected.py")
    resumed = fm.load_module(checkpoint)
    source = checkpoint.read_text(encoding="utf-8")
    assert resumed.semantic_hash == selected.semantic_hash
    assert "T.shared_workspace_descriptor(" in source
    assert "T.transfer_pipeline_channel(" in source
    assert "T.transfer_pipeline_contract(" in source


def test_selection_rejects_out_of_range_and_non_read_only_transfer_sources():
    module = semantic_packed_qkv_module()
    target = _target(source_index=2)
    with pytest.raises(IRVerificationError, match="invalid source operand 2"):
        _policy().apply(_policy().propose(module, target), target)

    function = module.prim_function_map["packed_qkv"]
    dispatch = kernel_dispatch_of(function)
    mutated_dispatch = replace(
        dispatch,
        writes=(*dispatch.writes, "fused_weight"),
    )
    mutated = replace(
        module,
        prim_functions=(replace(
            function,
            body=fm.T.sequential((mutated_dispatch,)),
        ),),
    )
    target = _target(source_index=1)
    with pytest.raises(IRVerificationError, match="non-read-only"):
        _policy().apply(_policy().propose(mutated, target), target)


def test_apply_rejects_a_missing_selection_for_a_recognized_semantic_op():
    target = _target()
    proposed = _policy().propose(semantic_packed_qkv_module(), target)
    edited = replace(
        proposed,
        selections=tuple(
            value for value in proposed.selections
            if value.point_id != "microkernel.packed_qkv"
        ),
    )

    with pytest.raises(IRVerificationError, match="has no selection record"):
        _policy().apply(edited, target)


def test_implementation_snapshot_includes_typed_workspace_contract():
    target = _target()
    first = dict(target.triton_implementation_model.snapshot())
    modified = replace(
        _implementation(),
        shared_workspaces=(fm.T.shared_workspace_descriptor(
            "rhs_stage", fm.tensor_type("bfloat16", (2, 128, 16)), 16
        ),),
        transfer_pipeline=fm.T.transfer_pipeline_contract((
            fm.T.transfer_pipeline_channel(
                "rhs", source_argument_indices=(1,), shared_workspace_indices=(0,)
            ),
        )),
    )
    second = dict(TritonImplementationModel(
        (modified,),
        {"qkv_parallel_linear": (modified.id,)},
        "test-workspace-machine/v1",
    ).snapshot())

    assert first["schema"] == "flagmega.triton-implementation-model/v3"
    assert first["sha256"] != second["sha256"]
