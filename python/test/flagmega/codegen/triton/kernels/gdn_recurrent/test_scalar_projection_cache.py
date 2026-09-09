# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A proved uniform head key must not require a CTA-wide cache-key vote."""

import ast

import pytest

from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry


def render_cache(uniform):
    call = dict(
        family="gdn_recurrent",
        variant="persistent",
        symbol="recurrent",
        signature="qkv, state",
        value_tile=8,
        local_capacity=256,
        active_values="active_extent",
        writer_active="writer",
        global_value="local_values + shard_index * 256",
        global_projection_value="recurrent_start + shard_index * 256",
        value_dim=128,
        projection_input="source",
        b_weight="b_weight",
        a_weight="a_weight",
        projection_dim=2048,
        projection_tile=1024,
        uniform_projection_rows=uniform,
        qkv="qkv",
        state="state",
        a_log="a_log",
        dt_bias="dt_bias",
        core_scratch="scratch",
        value_heads=32,
        key_heads=16,
        key_dim=128,
        repeats=2,
        head_block=128,
        qk_norm_add=True,
        qk_norm_epsilon=1e-6,
        round_normalized_qk=True,
        round_beta=True,
        round_core=True,
        z="z",
        result="output",
        norm_weight="norm_weight",
        z_offset="local_values",
        result_offset="local_values",
        epsilon=1e-6,
    )
    return TritonTemplateRegistry().render_kernel(
        KernelTemplateSpec("gdn_recurrent", "persistent", "nvidia", "sm90"),
        {
            "recurrent_value_tile": 8, "head_block": 128, "query_scale_repr": "0.125", "distributed_entry": False,
            "mesh_hierarchy": (1, 1), "render_calls": [call]
        },
    ).source


@pytest.mark.parametrize("uniform", [True, False])
def test_cache_key_shape_and_vote_follow_uniformity_proof(uniform):
    functions = {node.name: node for node in ast.parse(render_cache(uniform)).body if isinstance(node, ast.FunctionDef)}
    function = functions["recurrent"]
    initial = next(node.value for node in function.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "cached_heads" for target in node.targets))
    assert ast.literal_eval(initial.args[0]) == (() if uniform else (8, ))
    cache_update = next(node for node in ast.walk(function) if isinstance(node, ast.If))
    votes = [
        node for node in ast.walk(cache_update.test)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "sum"
    ]
    assert bool(votes) is not uniform
    if uniform:
        projections = [
            node for node in ast.walk(cache_update)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_flagmega_dense_rows"
        ]
        assert len(projections) == 2
        # Scalar rows are already one row: no subsequent min/any reduction.
        assert all(ast.literal_eval(call.args[-1]) is False for call in projections)
