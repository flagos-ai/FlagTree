# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Head projections are invariant across value tiles of the same head."""

import ast

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def test_recurrent_state_tile_consumes_cached_head_gates():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {"recurrent_value_tile": 4, "head_block": 128, "query_scale_repr": "0.125"},
    ).source
    tree = ast.parse(source)
    core = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_flagmega_gdn_recurrent_local_core_tile")
    # Recomputing A/B inside every state tile duplicates the K reduction even
    # when its head IDs and read-only projection input have not changed.
    calls = [node.func.id for node in ast.walk(core) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "_flagmega_dense_rows" not in calls
    arguments = {node.arg for node in core.args.args}
    assert {"beta", "decay"} <= arguments


def test_dense_rows_can_reduce_one_proven_uniform_head():
    source = TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {"recurrent_value_tile": 8, "head_block": 128, "query_scale_repr": "0.125"},
    ).source
    tree = ast.parse(source)
    projection = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_flagmega_dense_rows")
    assert "UNIFORM_ROWS" in {argument.arg for argument in projection.args.args}
    assert any(isinstance(node, ast.If) and isinstance(node.test, ast.Name)
               and node.test.id == "UNIFORM_ROWS" for node in ast.walk(projection))
