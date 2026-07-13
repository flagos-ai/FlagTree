# SPDX-License-Identifier: MIT
"""CTT-1 负例：非法嵌套、缺 ``end`` 时编译失败（与 B-stub 一致）。

契约 ID: **CTT-1**（负例）；依赖 ``triton.compile``（Ascend target + torch）。
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

_unit = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_module_a_doc", _unit / "_module_a_doc.py")
_mad = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mad)
__doc__ = _mad.extend_doc(__doc__)

pytest.importorskip("torch")

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

_ASCEND_TARGET = GPUTarget("npu", "Ascend910B", 0)


def _ast_source(fn):
    kwargs = {"fn": fn, "signature": {"x_ptr": "*fp32"}}
    if "constexprs" in inspect.signature(ASTSource).parameters:
        kwargs["constexprs"] = {}
    else:
        kwargs["constants"] = {}
    return ASTSource(**kwargs)


@pytest.mark.module_a
@pytest.mark.module_a_a1
def test_module_a_CTT1_illegal_nesting_compile_fails(fresh_triton_cache, capfd):
    _ = fresh_triton_cache

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        tl.debug_collect_start(level=1)
        tl.debug_collect_end()
        tl.debug_collect_end()

    src = _ast_source(kernel)
    with pytest.raises(Exception, match="PassManager::run failed"):
        triton.compile(src, target=_ASCEND_TARGET)
    assert "illegal nested debug collect region" in capfd.readouterr().err


@pytest.mark.module_a
@pytest.mark.module_a_a1
def test_module_a_CTT1_missing_end_compile_fails(fresh_triton_cache, capfd):
    _ = fresh_triton_cache

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        tl.store(x_ptr + tl.arange(0, 1), tl.zeros([1], dtype=tl.float32))

    src = _ast_source(kernel)
    with pytest.raises(Exception, match="PassManager::run failed"):
        triton.compile(src, target=_ASCEND_TARGET)
    assert "debug collect_begin without matching collect_end" in capfd.readouterr().err
