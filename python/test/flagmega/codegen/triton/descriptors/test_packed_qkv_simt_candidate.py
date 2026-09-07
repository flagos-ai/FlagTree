# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""SIMT QKV profile proofs, independent of model import and GPU execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.microkernels import (
    PackedQKVMicroKernelProvider, TIRMicroKernelContext,
)
from triton.flagmega.targets.nvidia import sm90_triton_implementation_model


IMPLEMENTATION = "tir.qkv_parallel_linear.packed_gemv_smem_pipeline"


def _context(module, changes=None):
    function = next(function for function in module.kernel_definitions
                    if (dispatch := fm.kernel_dispatch_of(function)) is not None
                    and dispatch.semantic_op == "ntt.packed_qkv_parallel_linear_fused_rhs")
    parameters = {name: replace(parameter, type=(changes or {}).get(name, parameter.type))
                  for name, parameter in function.parameter_map.items()}
    return TIRMicroKernelContext(module, SimpleNamespace(parameter_map=parameters),
                                fm.kernel_dispatch_of(function), sm90_triton_implementation_model())


def _ids(context):
    return {candidate.id for candidate in PackedQKVMicroKernelProvider().propose(context).candidates}


def test_simt_qkv_is_a_separate_non_matrix_implementation(packed_qkv_simt_pipeline_module):
    context = _context(packed_qkv_simt_pipeline_module)
    assert IMPLEMENTATION in _ids(context)
    implementation = context.implementation_model.implementation(IMPLEMENTATION)
    assert set(implementation.requires) == {"tma", "warp_specialize"}
    assert implementation.transfer_pipeline.capacity == 2


@pytest.mark.parametrize("index", (0, 1, 2))
def test_simt_qkv_proves_every_output_dtype(packed_qkv_simt_pipeline_module, index):
    context = _context(packed_qkv_simt_pipeline_module)
    name = context.dispatch.outputs[index]
    original = context.function.parameter_map[name].type
    different_dtype = replace(original, tensor=replace(
        original.tensor, dtype=fm.vector_type("float32", (8,)),
    ))
    assert IMPLEMENTATION not in _ids(_context(packed_qkv_simt_pipeline_module, {name: different_dtype}))


@pytest.mark.parametrize("shape", ((1, 1024), (1, 4096), (2, 2048)))
def test_simt_qkv_rejects_other_local_input_profiles(packed_qkv_simt_pipeline_module, shape):
    context = _context(packed_qkv_simt_pipeline_module)
    name = context.dispatch.arguments[0]
    original = context.function.parameter_map[name].type
    changed = replace(original, tensor=fm.tensor_type("bfloat16", shape))
    assert IMPLEMENTATION not in _ids(_context(packed_qkv_simt_pipeline_module, {name: changed}))


def test_simt_qkv_rejects_mismatched_weight_owners(packed_qkv_simt_pipeline_module):
    context = _context(packed_qkv_simt_pipeline_module)
    name = context.dispatch.arguments[1]
    original = context.function.parameter_map[name].type
    assert isinstance(original, fm.TensorType)
    changed = fm.tensor_type(original.dtype, (64, *original.shape[1:]))
    assert IMPLEMENTATION not in _ids(_context(packed_qkv_simt_pipeline_module, {name: changed}))


def test_simt_qkv_does_not_accept_split_k_partial_profile(packed_qkv_mma_pipeline_module):
    assert IMPLEMENTATION not in _ids(_context(packed_qkv_mma_pipeline_module))
