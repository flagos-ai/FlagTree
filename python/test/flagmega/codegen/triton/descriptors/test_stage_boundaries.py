# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tile expressions inline within, not across, an op's role boundary."""

import ast

import pytest

from triton.flagmega.codegen.triton.tir_package import describe_tir_package, render_tir_package


@pytest.mark.parametrize("fixture", [
    "packed_descriptor_pipeline_module",
    "packed_glu_direct_lhs_descriptor_pipeline_module",
    "packed_glu_paired_full_lhs_descriptor_pipeline_module",
    "packed_norm_stats_descriptor_pipeline_module",
    "packed_qkv_mma_pipeline_module",
])
def test_tile_helper_inline_and_op_roles_noinline(request, fixture):
    package = describe_tir_package(request.getfixturevalue(fixture))
    tree = ast.parse(render_tir_package(package, "unit"))
    definitions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    helpers = [name for name in definitions if name.endswith("__consumer_stage")]
    assert helpers
    for name in helpers:
        assert ast.unparse(definitions[name].decorator_list[0]) == "triton.jit"
        op = name.removesuffix("__consumer_stage")
        for role in ("producer", "consumer"):
            assert ast.unparse(
                definitions[f"{op}__{role}"].decorator_list[0]
            ) == "triton.jit(noinline=True)"
