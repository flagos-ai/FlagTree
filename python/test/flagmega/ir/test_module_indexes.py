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


@pytest.mark.parametrize("name", [
    "function_map",
    "prim_function_map",
    "kernel_callable_map",
    "execution_function_map",
])
def test_every_callable_index_is_shared_and_readonly(name):
    """A per-node lookup must not rebuild the whole callable table.

    Bufferize resolves a callee for every kernel call; when these indexes were
    plain properties, planning the 45-layer model spent minutes rebuilding
    them.  Sharing them is only sound because the module is immutable, so the
    index must also refuse writes.
    """

    module = _module()
    index = getattr(module, name)
    assert getattr(module, name) is index
    with pytest.raises(TypeError):
        index["main"] = None


def test_an_edited_module_rebuilds_its_function_index():
    module = _module()
    previous = module.function_map
    renamed = replace(module.functions[0], name="other")
    edited = replace(module, functions=(renamed, ), entry="other")
    assert "other" in edited.function_map
    assert "other" not in previous
