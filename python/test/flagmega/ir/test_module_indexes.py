# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""IR indexes share the immutable module's lifetime and edit boundary."""

from dataclasses import replace

import pytest

from triton.flagmega import ir as fm


def _module():
    node = fm.Node("value", "builtin.var", (), fm.tensor_type("float32", (8,)), attrs={"name": "value"})
    return fm.IRModule("high_level", "imported", (node,), (fm.Function("main", (node.id,), (node.id,)),), "main")


def test_node_lookup_reuses_readonly_index():
    module = _module()
    index = module.node_map
    assert module.node_map is index
    with pytest.raises(TypeError):
        index["value"] = replace(module.nodes[0], type=fm.tensor_type("float32", (16,)))


def test_replacing_or_reloading_module_cannot_reuse_stale_index():
    module = _module()
    previous = module.node_map
    replacement = replace(module.nodes[0], type=fm.tensor_type("float32", (16,)))
    edited = replace(module, nodes=(replacement,))
    loaded = fm.IRModule.from_data(module.semantic_data())
    assert edited.node_map["value"] is replacement
    assert previous["value"].type != replacement.type
    assert loaded.node_map is not previous
    assert loaded.node_map == previous
    assert loaded.node_map["value"] is not previous["value"]
