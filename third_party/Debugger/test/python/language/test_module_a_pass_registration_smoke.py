# SPDX-License-Identifier: MIT
"""模块 A：确认 ``passes.flagtree_debug`` 中 resolve / assign / instrumentation 可从 Python 引用。

契约 ID: **§3.1 编译管线挂接**（轻量 smoke，不重复 pass 逻辑）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

import pytest

from triton._C.libtriton.passes import flagtree_debug as fd


@pytest.mark.module_a
def test_module_a_flagtree_debug_passes_callable():
    assert callable(fd.has_debug_collect_markers)
    assert callable(fd.insert_default_debug_collect_markers)
    assert callable(fd.assign_debug_collect_scope_ids_without_erase)
    assert callable(fd.add_resolve_debug_scope)
    assert callable(fd.add_assign_debug_op_id)
    assert callable(fd.add_insert_instrumentation)
