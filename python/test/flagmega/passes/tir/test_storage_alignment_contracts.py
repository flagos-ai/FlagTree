# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.passes.tir.bufferize.alignment import storage_alignment_requirements, transfer_source_alignment_requirements
from python.test.flagmega.codegen.triton.microkernels.helpers import StubTarget, semantic_packed_qkv_module


def _target(*, prefer_strict=True, supported=("async_matrix",)):
    target = StubTarget(supported=supported)
    implementations = []
    for implementation, alignment in zip(target.triton_implementation_model.implementations, (16, 128), strict=True):
        workspace = fm.T.shared_workspace_descriptor("rhs_stage", fm.tensor_type("bfloat16", (2, 64, 16)), 16)
        pipeline = fm.T.transfer_pipeline_contract((fm.T.transfer_pipeline_channel(
            "rhs", source_argument_indices=(1,), shared_workspace_indices=(0,), source_alignment_bytes=alignment),))
        implementations.append(replace(implementation, shared_workspaces=(workspace,), transfer_pipeline=pipeline))
    order = tuple(implementation.id for implementation in implementations)
    target.triton_implementation_model = TritonImplementationModel(
        tuple(implementations), {"qkv_parallel_linear": order[::-1] if prefer_strict else order}, "alignment-test")
    return target


def _with_weight_alignment(module, alignment):
    return replace(module, prim_functions=tuple(replace(function, parameters=tuple(
        replace(parameter, alignment_bytes=alignment) if parameter.name == "fused_weight" else parameter
        for parameter in function.parameters)) for function in module.prim_functions))


def test_planning_uses_compatible_interfaces_not_the_preferred_candidate(tmp_path):
    source = semantic_packed_qkv_module()
    low, high = (_target(prefer_strict=preference).plan_storage_alignments(source) for preference in (False, True))
    assert low == high
    assert low.selection_points == source.selection_points
    assert low.selections == source.selections
    assert low.prim_function_map["packed_qkv"].parameter_map["fused_weight"].alignment_bytes == 128
    assert storage_alignment_requirements(low)["fused_weight"] == 128
    assert fm.load_module(fm.emit_module(low, tmp_path / "planned.py")) == low


def test_unsupported_interfaces_do_not_inflate_the_alignment_contract():
    target = _target(supported=())
    planned = target.plan_storage_alignments(semantic_packed_qkv_module())
    assert planned.prim_function_map["packed_qkv"].parameter_map["fused_weight"].alignment_bytes == 16


def test_selection_only_consumes_the_declared_contract():
    target = _target()
    planned = _with_weight_alignment(target.plan_storage_alignments(semantic_packed_qkv_module()), 16)
    proposed = target.propose_microkernels(planned)
    assert proposed.prim_functions == planned.prim_functions
    [point] = proposed.selection_points
    assert tuple(candidate.id for candidate in point.candidates) == ("test.qkv.scalar",)
    selected = target.select_microkernels(proposed)
    assert selected.nodes == planned.nodes
    assert selected.prim_function_map["packed_qkv"].parameter_map["fused_weight"].alignment_bytes == 16


def test_selecting_a_weaker_implementation_does_not_weaken_storage():
    target = _target(prefer_strict=False)
    planned = target.plan_storage_alignments(semantic_packed_qkv_module())
    selected = target.select_microkernels(target.propose_microkernels(planned))
    assert selected.nodes == planned.nodes
    dispatch = fm.kernel_dispatch_of(selected.prim_function_map["packed_qkv"])
    assert dispatch.microkernel.implementation == "test.qkv.scalar"
    assert storage_alignment_requirements(selected)["fused_weight"] == 128
    assert transfer_source_alignment_requirements(selected)["fused_weight"] == 128


def test_edited_selection_cannot_strengthen_a_storage_contract():
    target = _target()
    proposed = target.propose_microkernels(target.plan_storage_alignments(semantic_packed_qkv_module()))
    incompatible = _with_weight_alignment(proposed, 16)
    with pytest.raises(IRVerificationError, match="exceeding.*storage contract"):
        target.select_microkernels(incompatible)


def test_ir_verifier_rejects_an_implementation_that_exceeds_its_contract():
    target = _target()
    selected = target.select_microkernels(target.propose_microkernels(
        target.plan_storage_alignments(semantic_packed_qkv_module())))
    with pytest.raises(IRVerificationError, match="exceeds the declared alignment"):
        fm.verify_module(_with_weight_alignment(selected, 16))


@pytest.mark.parametrize("alignment", [0, -16, 3, True])
def test_parameter_alignment_is_a_typed_power_of_two(alignment):
    with pytest.raises(IRSchemaError, match="alignment_bytes"):
        fm.T.prim_parameter("x", fm.tensor_type("float32", (8,)), alignment_bytes=alignment)
