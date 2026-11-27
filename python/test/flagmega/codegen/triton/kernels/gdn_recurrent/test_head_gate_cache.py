# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Head-only nonlinear gates share the projection cache's exact lifetime."""

import ast

import pytest

from python.test.flagmega.codegen.triton.kernels.gdn_recurrent.test_scalar_projection_cache import render_cache


@pytest.mark.parametrize("uniform", [True, False])
def test_head_gates_are_computed_only_when_the_projection_key_changes(uniform):
    functions = {node.name: node for node in ast.parse(render_cache(uniform)).body if isinstance(node, ast.FunctionDef)}
    function = functions["recurrent"]
    cache_update = next(node for node in ast.walk(function) if isinstance(node, ast.If))
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_flagmega_gdn_head_gates"
    ]
    assert len(calls) == 1, "beta/decay still recomputed for every value tile"
    assert calls[0] in tuple(ast.walk(cache_update))
    assert ast.unparse(calls[0].args[2]) == "value_heads"
    assert ast.unparse(calls[0].args[3]) == ("projection_active" if uniform else "recurrent_mask")
    for name in ("beta", "decay"):
        initial = next(node.value for node in function.body if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets))
        assert ast.literal_eval(initial.args[0]) == (() if uniform else (8, ))
    core = functions["_flagmega_gdn_recurrent_local_core_tile"]
    arguments = {node.arg for node in core.args.args}
    assert {"beta", "decay"} <= arguments
    assert not {"beta_projection", "a_projection", "a_log", "dt_bias"} & arguments
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"sigmoid", "exp", "log1p"} for node in ast.walk(core))
