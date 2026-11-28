# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import ir as fm, pattern_match as pm
from triton.flagmega.egraph import EGraph
from triton.flagmega.egraph.matcher import find_egraph_matches


def graph(length, shared=False):
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    x = builder.var("x", fm.tensor_type("float32", (8, )), id="x")
    current = x
    for index in range(length):
        current = builder.call("tensors.reshape", (current, ), x.type, attrs={"shape": (8, )}, id=f"view{index}")
    builder.function("main", (x, ), (current, x) if shared else (current, ))
    return fm.verify_module(builder.build(entry="main"))


@pytest.mark.parametrize("length", [0, 1, 5])
def test_unary_chain_captures_unbounded_path_in_both_matchers(length):
    module = graph(length)
    pattern = pm.is_unary_chain(pm.is_var(name="source"), pm.is_op("tensors.reshape").with_user_count(1), name="path")
    root = module.node_map[module.functions[0].outputs[0]]
    result = pm.try_match_root(root, pattern, module)
    assert result["source"].id == "x"
    assert [node.id for node in result["path"]] == [f"view{i}" for i in reversed(range(length))]
    egraph = EGraph()
    egraph.add_module(module)
    matches = find_egraph_matches(egraph, pattern, module)
    assert any(result.root.id == root.id and len(result["path"]) == length for result in matches)


def test_chain_does_not_cross_a_shared_step():
    from dataclasses import replace
    module = graph(2)
    function = replace(module.functions[0], outputs=("view1", "view0"))
    module = replace(module, functions=(function, ))
    pattern = pm.is_unary_chain(pm.is_var(), pm.is_op("tensors.reshape").with_user_count(1))
    assert pm.try_match_root(module.node_map["view1"], pattern, module) is None


def test_chain_does_not_swallow_a_different_unary_operator():
    from dataclasses import replace
    module = graph(2)
    module = replace(module,
                     nodes=tuple(replace(n, op="math.silu", attrs={}) if n.id == "view0" else n for n in module.nodes))
    pattern = pm.is_unary_chain(pm.is_var(), pm.is_op("tensors.reshape"))
    assert pm.try_match_root(module.node_map["view1"], pattern, module) is None
