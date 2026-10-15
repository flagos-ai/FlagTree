# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import import_module

from triton.flagmega import ir as fm


def _legacy_semantic_hash(module: fm.IRModule) -> str:
    payload = json.dumps(
        module.semantic_data(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_with_selection():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value = builder.var("value", fm.tensor_type("bfloat16", (1, 16)), id="value")
    builder.function("main", (value,), (value,))
    module = builder.build(entry="main")
    point = fm.SelectionPoint(
        "point",
        "unit",
        (fm.Candidate("candidate", {"tile": 16}, {"legal": True}),),
        "candidate",
        owner="value",
    )
    return replace(module, selection_points=(point,)), point


def test_fragment_encoder_preserves_legacy_semantic_hash_bytes():
    module, _ = _module_with_selection()

    assert module.semantic_hash == _legacy_semantic_hash(module)


def test_shared_tensor_types_are_encoded_once_in_a_large_node_sequence(monkeypatch):
    model = import_module("triton.flagmega.ir.model")
    value_type = fm.tensor_type("float32", (16,))
    nodes = tuple(fm.Node(f"x{i}", "builtin.var", (), value_type, attrs={"name": f"x{i}"}) for i in range(50))
    module = fm.IRModule("high_level", "imported", nodes, (fm.Function("main", (), (nodes[-1].id,)),), "main")
    expected = _legacy_semantic_hash(module)
    calls = []
    original = fm.TensorType.to_data

    def measure(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(fm.TensorType, "to_data", measure)
    model._SEMANTIC_COMPONENT_JSON.clear()
    assert module.semantic_hash == expected
    assert len(calls) == 1


def test_stage_only_changes_reuse_unchanged_sequence_encoding(monkeypatch):
    model = import_module("triton.flagmega.ir.model")
    module, _ = _module_with_selection()
    snapshots = [replace(module, stage=stage) for stage in ("imported", "packed", "distributed")]
    calls = []
    original = model._semantic_component_json

    def count(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(model, "_semantic_component_json", count)
    for snapshot in snapshots:
        assert snapshot.semantic_hash == _legacy_semantic_hash(snapshot)
    # Node/Function/SelectionPoint sequences do not change with the stage.
    sequence_ids = {id(value) for value in (*module.nodes, *module.functions, *module.selection_points)}
    assert len([value for value in calls if id(value) in sequence_ids]) == len(sequence_ids)


def test_equal_python_literals_with_different_json_bytes_keep_distinct_hashes():
    module, _ = _module_with_selection()
    first = replace(module, nodes=(replace(module.nodes[0], metadata={"flag": True}),))
    second = replace(first, nodes=(replace(first.nodes[0], metadata={"flag": 1}),))
    assert first.nodes == second.nodes  # Python equality cannot key this cache.
    assert first.semantic_hash == _legacy_semantic_hash(first)
    assert second.semantic_hash == _legacy_semantic_hash(second)
    assert first.semantic_hash != second.semantic_hash


def test_shared_immutable_selection_is_encoded_once_across_modules(monkeypatch):
    model = import_module("triton.flagmega.ir.model")
    module, point = _module_with_selection()
    second = replace(module, stage="canonical")
    original = fm.SelectionPoint.to_data
    calls = 0

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    model._SEMANTIC_COMPONENT_JSON.clear()
    monkeypatch.setattr(fm.SelectionPoint, "to_data", counted)
    try:
        assert module.semantic_hash == _legacy_semantic_hash(module)
        assert second.semantic_hash == _legacy_semantic_hash(second)
        assert calls == 3  # one cached encoder call plus two legacy controls
        assert model._SEMANTIC_COMPONENT_JSON[id(point)][0]() is point
    finally:
        model._SEMANTIC_COMPONENT_JSON.clear()


def test_module_replacement_reuses_recursively_frozen_metadata():
    builder = fm.IRBuilder(
        dialect="high_level",
        stage="imported",
        metadata={"nested": {"tiles": [8, 16]}},
    )
    value = builder.var("value", fm.tensor_type("bfloat16", (1, 16)), id="value")
    builder.function("main", (value,), (value,))
    module = builder.build(entry="main")

    replaced = replace(module, stage="canonical")

    assert replaced.metadata is module.metadata
    assert replaced.metadata["nested"] is module.metadata["nested"]
    assert replaced.metadata["nested"]["tiles"] == (8, 16)


def test_shared_frozen_metadata_keeps_exact_hash_and_encoded_fragment():
    model = import_module("triton.flagmega.ir.model")
    builder = fm.IRBuilder(
        dialect="high_level",
        stage="imported",
        metadata={"nested": {"tiles": [8, 16]}},
    )
    value = builder.var("value", fm.tensor_type("bfloat16", (1, 16)), id="value")
    builder.function("main", (value,), (value,))
    module = builder.build(entry="main")
    replaced = replace(module, stage="canonical")

    model._SEMANTIC_COMPONENT_JSON.clear()
    try:
        assert module.semantic_hash == _legacy_semantic_hash(module)
        assert replaced.semantic_hash == _legacy_semantic_hash(replaced)
        cached = model._SEMANTIC_COMPONENT_JSON[id(module.metadata)]
        assert cached[0]() is module.metadata
    finally:
        model._SEMANTIC_COMPONENT_JSON.clear()
