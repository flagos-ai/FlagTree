# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from triton.flagmega.codegen.triton.microkernels import (
    PackedQKVMicroKernelProvider,
    TIRMicroKernelContext,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import kernel_dispatch_of

from .helpers import implementation_model, semantic_packed_qkv_module


def _context(module):
    function = module.prim_function_map["packed_qkv"]
    dispatch = kernel_dispatch_of(function)
    assert dispatch is not None
    return TIRMicroKernelContext(module, function, dispatch, implementation_model())


def test_provider_consumes_only_canonical_semantics_and_injected_catalog():
    proposal = PackedQKVMicroKernelProvider().propose(
        _context(semantic_packed_qkv_module())
    )

    assert proposal is not None
    assert tuple(value.id for value in proposal.candidates) == (
        "test.qkv.scalar",
        "test.qkv.pipeline",
    )
    assert proposal.default_candidate == "test.qkv.pipeline"
    assert proposal.candidates[1].facts["requires"] == ("async_matrix",)


def test_provider_is_absent_before_fused_rhs_canonicalization():
    module = semantic_packed_qkv_module(op="ntt.packed_qkv_parallel_linear")
    assert PackedQKVMicroKernelProvider().op_names.isdisjoint(
        {kernel_dispatch_of(module.prim_functions[0]).semantic_op}
    )


def test_provider_rejects_malformed_semantic_contract():
    module = semantic_packed_qkv_module()
    function = module.prim_functions[0]
    dispatch = kernel_dispatch_of(function)
    malformed = replace(dispatch, semantic_attrs={
        **dict(dispatch.semantic_attrs),
        "projection_n_capacities": (2048, 0, 1024),
    })
    module = replace(
        module,
        prim_functions=(replace(
            function,
            body=replace(function.body, fields=(malformed,)),
        ),),
    )

    with pytest.raises(CodegenError, match="three positive"):
        PackedQKVMicroKernelProvider().propose(_context(module))


def test_portable_provider_contains_no_machine_policy_or_tile_constant():
    source = Path(
        inspect.getsourcefile(PackedQKVMicroKernelProvider) or ""
    ).read_text(encoding="utf-8").lower()

    assert "nvidia" not in source
    assert "sm90" not in source
    assert "block_n =" not in source
    assert "block_k =" not in source
    assert "mesh_size" not in source
