# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""An aligned alias offset also requires an aligned arena base."""

import ast
import pytest
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package


@pytest.mark.parametrize("fixture", ("packed_qkv_mma_pipeline_module", "packed_qkv_mma_aligned_pipeline_module"))
def test_shared_arena_base_preserves_all_workspace_alignments(request, fixture):
    package = describe_tir_package(request.getfixturevalue(fixture))
    schedule = package["pipeline_schedule"]
    expected = 2048 if "aligned" in fixture else 1024
    assert schedule["shared_arena_alignment_bytes"] == expected
    tree = ast.parse(render_tir_package(package, "unit"))
    arena = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(isinstance(value, ast.Name) and value.id == "_flagmega_shared_arena" for value in node.targets))
    assert next(ast.literal_eval(value.value) for value in arena.keywords if value.arg == "alignment_bytes") == expected
    assert "qkv_c_m ^" not in ast.unparse(tree)
