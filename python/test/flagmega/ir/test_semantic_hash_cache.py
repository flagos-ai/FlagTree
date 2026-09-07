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
