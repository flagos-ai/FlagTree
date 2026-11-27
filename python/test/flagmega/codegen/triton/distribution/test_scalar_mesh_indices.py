# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Scalar mesh coordinates do not need tensor-layout isolation roots."""

import ast

import pytest

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context


@pytest.mark.parametrize("hierarchy,levels", [
    ((2, 4), "bb"), ((2, 2, 2), "bbb"), ((1, 1, 8), "cdb"),
])
@pytest.mark.parametrize("distributed", [True, False])
def test_scalar_mesh_coordinates_are_direct_and_keep_int64(hierarchy, levels, distributed):
    context = _mesh_context({"hierarchy": hierarchy, "hierarchy_levels": levels,
                             "name": "abc"[:len(hierarchy)]})
    source = TritonTemplateRegistry().environment.from_string(
        '{% import "entrypoints/_common.py.jinja" as entry with context %}\n'
        'def coordinates():\n{{ entry.mesh_coordinates() }}\n'
    ).render(**context, distributed_entry=distributed)
    assert "rematerialize_index" not in source
    tree = ast.parse(source)
    assignments = {node.targets[0].id: node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)}
    for axis, level in enumerate(levels):
        value = assignments[f"shard_coord{axis}"]
        assert isinstance(value, ast.Call)
        if distributed and level == "b":
            assert value.func.attr == "to"
            assert ast.unparse(value.args[0]) == "tl.int64"
            assert ast.unparse(value.func.value.func) == "tle.shard_id"
        else:
            assert ast.unparse(value.func) == "tl.full"
            assert [ast.unparse(arg) for arg in value.args] == ["()", "0", "tl.int64"]
    index = ast.unparse(assignments["shard_index"])
    if distributed:
        expected = "shard_coord0"
        for axis, extent in enumerate(hierarchy[1:], 1):
            expected = f"({expected} * {extent} + shard_coord{axis})"
        assert index == ast.unparse(ast.parse(expected, mode="eval").body)
    else:
        assert index == "0"
