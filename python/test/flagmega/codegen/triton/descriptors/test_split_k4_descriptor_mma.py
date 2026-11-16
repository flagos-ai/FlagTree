# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit four-partition reduction order for the packed TMA consumer."""
from copy import deepcopy

import pytest

from triton.flagmega.codegen.triton.kernel_call_renderers import _dense_matmul_call
from triton.flagmega.codegen.triton.runtime_binding import describe_function_runtime_binding
from triton.flagmega.errors import CodegenError

ORDER = "split_k4_k32_mma_k16_pairtree"


@pytest.fixture(scope="module")
def split_raw():
    from .test_dense_lhs_staging import make_module
    binding = describe_function_runtime_binding(make_module())
    raw = deepcopy(next(c for c in binding["call_abi"]["kernel_calls"] if c["family"] == "dense_matmul"))
    raw["semantic_attrs"]["reduction_order"] = ORDER
    raw["parameters"].update(accumulation_mode="split_k4_mma", mma_block_k=128)
    raw["facts"]["split_k4_ordered_mma_accumulation"] = True
    return raw


def test_explicit_split_k4_contract_is_encoded(split_raw):
    call = _dense_matmul_call(deepcopy(split_raw))
    assert call["accumulation_mode"] == "split_k4_mma"
    assert call["mma_block_k"] == 128
    assert call["local_k_capacity"] == 6144
    assert call["lhs_stage_extent"] == 8192


@pytest.mark.parametrize("mutation", ["old_order", "relaxed_order", "simt", "old_mma", "other_variant", "missing_fact", "old_fact", "k64", "tile32", "warps4", "slots2", "no_stage", "rhs_option", "split_count", "fact_only"])
def test_split_k4_rejects_mismatched_contract(split_raw, mutation):
    raw = deepcopy(split_raw)
    if mutation == "old_order": raw["semantic_attrs"]["reduction_order"] = "mma_k16"
    elif mutation == "relaxed_order": raw["semantic_attrs"]["reduction_order"] = "relaxed"
    elif mutation == "simt": raw["parameters"]["accumulation_mode"] = "simt"
    elif mutation == "old_mma": raw["parameters"]["accumulation_mode"] = "ordered_mma"
    elif mutation == "other_variant": raw["variant"] = "packed_k_major_gemv"
    elif mutation == "missing_fact": raw["facts"].pop("split_k4_ordered_mma_accumulation")
    elif mutation == "old_fact": raw["facts"]["ordered_mma_accumulation"] = True
    elif mutation == "k64": raw["parameters"]["mma_block_k"] = 64
    elif mutation == "tile32": raw["parameters"]["tile_n"] = 32
    elif mutation == "warps4": raw["parameters"]["consumer_warps"] = 4
    elif mutation == "slots2": raw["transfer_pipeline"]["capacity"] = 2
    elif mutation == "no_stage": raw["parameters"].pop("lhs_stage_extent")
    elif mutation == "rhs_option": raw["parameters"]["mma_rhs_load"] = "direct_dot"
    elif mutation == "split_count": raw["parameters"]["accumulation_splits"] = 2
    elif mutation == "fact_only":
        raw["semantic_attrs"]["reduction_order"] = "relaxed"
        raw["parameters"]["accumulation_mode"] = "simt"
    with pytest.raises(CodegenError, match="[Ss]plit"):
        _dense_matmul_call(raw)


@pytest.mark.parametrize("norm_stats", (False, True))
def test_split_helper_two_includes_preserve_one_terminal_tree(norm_stats):
    from .test_dense_lhs_staging import make_module, direct_norm_stats_staged_module
    from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
    module = direct_norm_stats_staged_module() if norm_stats else make_module()
    package = describe_tir_package(module)
    call = next(c for c in package["render_calls"] if c["family"] == "dense_matmul")
    call.update(accumulation_mode="split_k4_mma", mma_block_k=128)
    source = render_tir_package(package, "split-k4-test")
    assert "[2, 0], [4, 1, 2], [1, 16, 8]" in source
    assert "[4, 32, 16]" in source and "[4, 16, 32]" in source
    assert "dense_accumulator = tl.dot(dense_a, dense_b, dense_accumulator)" in source
    assert "(dense_p0 + dense_p1) + (dense_p2 + dense_p3)" in source
    assert source.count("__split_k4_finalize(dense_accumulator)") == 2  # definition + terminal call
    if norm_stats:
        assert source.index("__split_k4_finalize(dense_accumulator)", source.index("def "+call["symbol"]+"__consumer(")) < source.index("dense_projection = dense_accumulator.to(")
        raw = deepcopy(next(c for c in describe_function_runtime_binding(module)["call_abi"]["kernel_calls"] if c["family"] == "dense_matmul"))
        raw["semantic_attrs"]["reduction_order"] = ORDER
        raw["parameters"].update(accumulation_mode="split_k4_mma", mma_block_k=128)
        raw["facts"]["split_k4_ordered_mma_accumulation"] = True
        # This older neighbour uses synchronous staging. The fixed new
        # implementation must reject it, while its shared template preserves
        # the projection-rounding/epilogue order for a matching typed provider.
        assert raw["parameters"].get("lhs_copy_kind", "sync") == "sync"
        with pytest.raises(CodegenError, match="Split-K4"):
            _dense_matmul_call(raw)
