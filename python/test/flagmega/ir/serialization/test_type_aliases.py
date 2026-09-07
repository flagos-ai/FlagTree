# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Editable Python checkpoints intern repeated immutable IR types."""

import gc
from dataclasses import replace
from weakref import ref

import pytest

import triton.flagmega.ir.python_ir as python_ir
from triton.flagmega import ir as fm


def _module_with_repeated_distributed_type():
    value_type = fm.DistributedType(
        fm.tensor_type(fm.vector_type("bfloat16", (8,)), (256, 32)),
        (
            fm.SBP.split(fm.SplitStage.contiguous((0,), 32)),
            fm.SBP.broadcast(),
        ),
        fm.Placement((8, 16), "yx", "bb"),
    )
    builder = fm.IRBuilder(dialect="high_level", stage="distributed")
    lhs = builder.var("lhs", value_type, id="lhs")
    rhs = builder.var("rhs", value_type, id="rhs")
    builder.function("main", (lhs, rhs), (lhs, rhs))
    return builder.build(entry="main")


def test_repeated_types_are_named_once_and_remain_editable_python(tmp_path):
    module = _module_with_repeated_distributed_type()
    source = fm.module_source(module)

    assert "TYPE_0000 = fm.DistributedType(" in source
    assert source.count("fm.DistributedType(") == 1
    assert source.count("TYPE_0000") == 3  # definition plus two input uses

    checkpoint = fm.emit_module(module, tmp_path / "repeated_types.py")
    restored = fm.load_module(checkpoint)
    assert restored.semantic_hash == module.semantic_hash
    assert restored.node_map["lhs"].type is restored.node_map["rhs"].type


def test_single_use_type_stays_inline_for_local_editability():
    builder = fm.IRBuilder(dialect="high_level", stage="imported")
    value = builder.var("value", fm.tensor_type("float32", (3,)), id="value")
    builder.function("main", (value,), (value,))

    source = fm.module_source(builder.build(entry="main"))

    assert "TYPE_0000" not in source
    assert "fm.tensor_type(" in source


def test_shared_fragment_cache_observes_an_edited_node_type():
    original = _module_with_repeated_distributed_type()
    assert "TYPE_0000" in fm.module_source(original)

    edited_nodes = tuple(
        replace(node, type=fm.tensor_type("float32", (7,)))
        if node.id == "rhs"
        else node
        for node in original.nodes
    )
    edited = replace(original, nodes=edited_nodes)
    source = fm.module_source(edited)

    assert "TYPE_0000" not in source
    assert "fm.DistributedType(" in source
    assert "'float32'" in source
    assert "[7]" in source


def test_shared_expression_cache_reuses_only_the_same_selection_object(monkeypatch):
    base = _module_with_repeated_distributed_type()
    point = fm.SelectionPoint(
        id="choice",
        kind="unit-test",
        candidates=(fm.Candidate("keep", {"tile": 16}, {}),),
        default_candidate="keep",
    )
    original = replace(base, selection_points=(point,))
    fm.module_source(original)

    def fail_if_reexpanded(_value):
        raise AssertionError("shared selection point was expanded again")

    monkeypatch.setattr(python_ir, "_candidate_expr", fail_if_reexpanded)
    stage_edit = replace(original, stage="packed")
    assert "STAGE = 'packed'" in fm.module_source(stage_edit)

    edited_point = replace(
        point,
        candidates=(fm.Candidate("keep", {"tile": 32}, {}),),
    )
    with pytest.raises(AssertionError, match="expanded again"):
        fm.module_source(replace(stage_edit, selection_points=(edited_point,)))


def test_expression_fragments_do_not_retain_ir_owners():
    def populate_cache():
        module = _module_with_repeated_distributed_type()
        point = fm.SelectionPoint(
            id="choice",
            kind="unit-test",
            candidates=(fm.Candidate("keep", {"tile": 16}, {}),),
            default_candidate="keep",
        )
        module = replace(module, selection_points=(point,))
        fm.module_source(module)
        return ref(module), ref(point)

    module_ref, point_ref = populate_cache()
    gc.collect()

    assert module_ref() is None
    assert point_ref() is None
