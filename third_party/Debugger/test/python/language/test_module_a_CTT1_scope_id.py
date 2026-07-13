# SPDX-License-Identifier: MIT
"""CTT-1 正例：``assign_debug_collect_scope_ids_without_erase`` 写入 ``scope_id``（可观测 IR）。

契约 ID: **CTT-1**（正例）；纯 MLIR 绑定，无需 GPU。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

import triton  # noqa: F401
from triton._C.libtriton import ir
from triton._C.libtriton.passes import flagtree_debug as fd


def _parse_mlir(tmp_path: Path, text: str):
    ctx = ir.context()
    ir.load_dialects(ctx)
    path = tmp_path / "module_a_ctt1.mlir"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    mod = ir.parse_mlir_module(str(path), ctx)
    mod._flagtree_keepalive_context = ctx
    return mod


@pytest.mark.module_a
@pytest.mark.module_a_ctt1
def test_module_a_CTT1_scope_id_assigned_in_ir(tmp_path):
    mod = _parse_mlir(
        tmp_path,
        """
        module {
          "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
          "flagtree_debug.collect_end"() : () -> ()
        }
        """,
    )

    assert fd.assign_debug_collect_scope_ids_without_erase(mod) is True
    text = mod.str()
    assert "scope_id" in text
    assert "flagtree_debug.collect_begin" in text
    assert "flagtree_debug.collect_end" in text


@pytest.mark.module_a
@pytest.mark.module_a_ctt1
def test_module_a_CTT1_assign_fails_on_illegal_nesting_without_erase(tmp_path):
    mod = _parse_mlir(
        tmp_path,
        """
        module {
          "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
          "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
          "flagtree_debug.collect_end"() : () -> ()
          "flagtree_debug.collect_end"() : () -> ()
        }
        """,
    )

    assert fd.assign_debug_collect_scope_ids_without_erase(mod) is False
