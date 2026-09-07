# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Profile legality coverage for the packed BF16 QKV MMA candidate."""

from dataclasses import replace
from types import SimpleNamespace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels import (
    PackedQKVMicroKernelProvider,
    TIRMicroKernelContext,
)
from triton.flagmega.ir import kernel_dispatch_of
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


_CANDIDATE = "tir.qkv_parallel_linear.packed_partial_mma_smem_pipeline"
_TABLE_CANDIDATE = (
    "tir.qkv_parallel_linear."
    "packed_partial_mma_descriptor_table_smem_pipeline"
)


def _context(module, *, parameter_types=None):
    function = next(
        value
        for value in module.kernel_definitions
        if (dispatch := kernel_dispatch_of(value)) is not None
        and dispatch.semantic_op
        == "ntt.packed_qkv_parallel_linear_fused_rhs"
    )
    dispatch = kernel_dispatch_of(function)
    assert dispatch is not None
    parameters = function.parameter_map
    if parameter_types:
        parameters = {
            name: replace(parameter, type=parameter_types.get(name, parameter.type))
            for name, parameter in parameters.items()
        }
    return TIRMicroKernelContext(
        module,
        SimpleNamespace(parameter_map=parameters),
        dispatch,
        sm90_triton_implementation_model(),
    )


def _candidate_ids(context):
    proposal = PackedQKVMicroKernelProvider().propose(context)
    assert proposal is not None
    return tuple(value.id for value in proposal.candidates)


def test_packed_qkv_mma_is_selected_for_its_exact_local_profile(
    packed_qkv_mma_pipeline_module,
):
    ids = _candidate_ids(_context(packed_qkv_mma_pipeline_module))

    assert ids[0] == _CANDIDATE
    assert _TABLE_CANDIDATE in ids


def test_packed_qkv_mma_rejects_wrong_reduction_extent_and_multiple_rows(
    packed_qkv_mma_pipeline_module,
):
    base = _context(packed_qkv_mma_pipeline_module)
    source = base.function.parameter_map[base.dispatch.arguments[0]].type
    assert isinstance(source, fm.DistributedType)

    for shape in ((1, 1536), (2, 2048)):
        modified = replace(source, tensor=fm.tensor_type("bfloat16", shape))
        ids = _candidate_ids(_context(
            packed_qkv_mma_pipeline_module,
            parameter_types={base.dispatch.arguments[0]: modified},
        ))
        assert _CANDIDATE not in ids


def test_packed_qkv_partial_mma_requires_sum_partial_on_input_split_axes(
    packed_qkv_mma_pipeline_module,
):
    base = _context(packed_qkv_mma_pipeline_module)
    output_names = base.dispatch.outputs
    output_types = tuple(
        base.function.parameter_map[name].type for name in output_names
    )
    assert all(isinstance(value, fm.DistributedType) for value in output_types)
    placement = output_types[0].placement
    wrong_partial_types = {
        name: replace(
            value,
            partial=fm.SBP.partial((1,), fm.ReduceOp.SUM),
        )
        for name, value in zip(output_names, output_types, strict=True)
    }

    ids = _candidate_ids(_context(
        packed_qkv_mma_pipeline_module,
        parameter_types=wrong_partial_types,
    ))

    assert _CANDIDATE not in ids


def test_packed_qkv_partial_mma_rejects_non_sum_partial(
    packed_qkv_mma_pipeline_module,
):
    base = _context(packed_qkv_mma_pipeline_module)
    wrong = {
        name: replace(
            base.function.parameter_map[name].type,
            partial=fm.SBP.partial((0,), fm.ReduceOp.MAX),
        )
        for name in base.dispatch.outputs
    }

    ids = _candidate_ids(_context(
        packed_qkv_mma_pipeline_module,
        parameter_types=wrong,
    ))

    assert _CANDIDATE not in ids
