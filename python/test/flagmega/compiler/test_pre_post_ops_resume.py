# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from python.test.flagmega.ir.fusion.test_pre_post_ops import make_graph


def test_edit_fusion_python_and_resume_before_tir(tmp_path):
    from dataclasses import replace
    source = replace(make_graph(), stage="frozen_constants")
    path = fm.emit_module(source, tmp_path / "before.py")
    edited = path.read_text().replace("F.nn.softmax,", "F.math.sigmoid,").replace("axis=-1,", "")
    path.write_text(edited)
    changed = fm.load_module(path)
    assert changed.semantic_hash != source.semantic_hash
    result = Compiler().compile(changed).module
    dispatches = [value.dispatch for value in result.kernel_definitions]
    assert any(value.semantic_op == "math.sigmoid" and value.semantic_attrs.get("pre_ops") for value in dispatches)
    checkpoint = fm.emit_module(result, tmp_path / "after.py")
    assert fm.load_module(checkpoint).semantic_hash == result.semantic_hash


def test_authored_fusion_is_not_lost_by_packing_or_distribution():
    source = make_graph()
    result = Compiler().compile(source).module
    dispatches = [value.dispatch for value in result.kernel_definitions]
    softmax = [value for value in dispatches if value.semantic_op == "nn.softmax"]
    assert softmax
    assert any(value.semantic_attrs.get("pre_ops") and value.semantic_attrs.get("post_ops") for value in softmax)
