# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A partial-order implementation cannot satisfy an implicit or full-K order."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from triton.flagmega.codegen.triton.candidates.dense_matmul import _supports_reduction_order
from triton.flagmega.codegen.triton.microkernels import PackedQKVMicroKernelProvider
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import kernel_dispatch_of
from python.test.flagmega.codegen.triton.microkernels.test_packed_qkv_provider import _context
from python.test.flagmega.codegen.triton.microkernels.helpers import semantic_packed_qkv_module


ORDER = "split_k4_k32_mma_k16_pairtree"


def implementation(kind):
    facts, parameters = {}, {}
    if kind == "old":
        facts["ordered_mma_accumulation"] = True
    if kind in {"new", "mixed", "missing_mode"}:
        facts["split_k4_ordered_mma_accumulation"] = True
        if kind != "missing_mode":
            parameters["accumulation_mode"] = "split_k4_mma"
        if kind == "mixed":
            facts["ordered_mma_accumulation"] = True
    if kind == "missing_fact":
        parameters["accumulation_mode"] = "split_k4_mma"
    return SimpleNamespace(facts=facts, parameters=parameters)


@pytest.mark.parametrize("kind", ["simt", "old", "new", "mixed", "missing_mode", "missing_fact"])
@pytest.mark.parametrize("order", [None, "relaxed", "mma_k16", ORDER, "unknown"])
def test_order_and_implementation_match_both_directions(kind, order):
    attrs = {} if order is None else {"reduction_order": order}
    expected = (kind in {"simt", "old"} and order in {None, "relaxed"}
                or kind == "old" and order == "mma_k16"
                or kind == "new" and order == ORDER)
    assert _supports_reduction_order(implementation(kind), attrs) is expected


def test_full_k_order_still_rejects_accumulation_splits():
    old = implementation("old")
    old.parameters["accumulation_splits"] = 2
    assert not _supports_reduction_order(old, {"reduction_order": "mma_k16"})


def test_direct_tir_packed_qkv_rejects_new_order_before_implementation_selection():
    module = semantic_packed_qkv_module()
    function = module.prim_functions[0]
    dispatch = kernel_dispatch_of(function)
    changed = replace(dispatch, semantic_attrs={**dispatch.semantic_attrs, "reduction_order": ORDER})
    module = replace(module, prim_functions=(replace(function, body=replace(function.body, fields=(changed,))),))
    with pytest.raises(CodegenError, match="reduction_order"):
        PackedQKVMicroKernelProvider().propose(_context(module))
