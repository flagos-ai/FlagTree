# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Matching costs must scale with graph snapshots, not candidate roots."""

import gc
import weakref
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega import pattern_match as pm
from triton.flagmega.pattern_match import matcher


def _module():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    tensor = fm.tensor_type("float32", (1, 8))
    value = builder.var("value", tensor, id="value")
    activated = builder.call("math.silu", (value,), tensor, id="activated")
    output = builder.call("math.add", (activated, activated), tensor, id="output")
    builder.function("main", (value,), (output,))
    return builder.build(entry="main")


def _count_analysis(monkeypatch):
    calls = []
    original = matcher._user_counts

    def count(module):
        calls.append(id(module))
        return original(module)

    monkeypatch.setattr(matcher, "_user_counts", count)
    return calls


def test_plain_pattern_does_not_scan_graph_users(monkeypatch):
    calls = _count_analysis(monkeypatch)
    module = _module()
    assert len(pm.find_matches(module, pm.F.math.is_silu())) == 1
    assert calls == []


def test_user_analysis_is_shared_across_roots_and_or_branches(monkeypatch):
    calls = _count_analysis(monkeypatch)
    module = _module()
    pattern = pm.F.math.is_silu().with_user_count(2) | pm.F.math.is_silu().with_user_count(1)
    for _ in range(3):
        assert len(pm.find_matches(module, pattern)) == 1
    assert calls == [id(module)]


def test_changed_outputs_and_equal_reloaded_modules_get_fresh_users(monkeypatch):
    calls = _count_analysis(monkeypatch)
    module = _module()
    pattern = pm.F.math.is_silu().with_user_count(1)
    node = module.node_map["activated"]
    assert pm.try_match_root(node, pattern, module)
    # Function output edges count separately; repeated SSA inputs count once.
    edited = replace(module, functions=(replace(module.functions[0], outputs=("output", "activated")),))
    assert pm.try_match_root(node, pattern, edited) is None
    loaded = fm.IRModule.from_data(module.semantic_data())
    assert pm.try_match_root(loaded.node_map[node.id], pattern, loaded)
    assert calls == [id(module), id(edited), id(loaded)]


def test_cached_analysis_does_not_retain_module():
    module = _module()
    assert pm.find_matches(module, pm.wildcard().with_user_count(1))
    reference = weakref.ref(module)
    del module
    gc.collect()
    assert reference() is None
