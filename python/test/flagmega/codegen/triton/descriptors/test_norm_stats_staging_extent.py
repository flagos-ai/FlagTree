# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Bounded LHS capacity does not imply initializing unused Shared padding."""

import pytest

from .conftest import _packed_norm_stats_pipeline_module
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


@pytest.mark.parametrize("k", (1024, 2048, 4096, 6144, 8192))
def test_norm_stats_stages_only_the_logical_local_reduction(k):
    module = _packed_norm_stats_pipeline_module(
        "tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_norm_stats_lhs8192",
        reduction_extent=k,
    )
    package = describe_tir_package(module)
    call = next(value for value in package["render_calls"] if value["semantic_op"] == "ntt.matmul_norm_stats")
    assert call["local_k_capacity"] == k
    assert call["shared_workspaces"][1]["shape"] == (1, 8192)
    source = render_tir_package(package, "unit")
    assert f"for dense_copy_start in tl.range(0, {k}, {call['lhs_copy_tile']})" in source
    assert f"mask=(dense_local_k_offsets < {k})" in source
