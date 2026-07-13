# SPDX-License-Identifier: MIT
"""A-1: TTIR 持久化后 marker 被擦除；``debug_enabled`` 与 collect 一致。

契约 ID: **A-1**；依赖 ``triton.compile``（Ascend target + torch）。
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
def test_module_a_A1_markers_absent_from_persisted_ttir(fresh_triton_cache):
    """ResolveDebugScopePass stub 擦除 marker；方言名为 ``flagtree_debug``（非旧稿 ``flagtree.debug``）。"""

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        tl.debug_collect_end()

    _ = fresh_triton_cache
    src = _ast_source(kernel)
    out = triton.compile(src, target=_ASCEND_TARGET)
    ttir = out.asm["ttir"]
    assert "flagtree_debug.collect_begin" not in ttir
    assert "flagtree_debug.collect_end" not in ttir
    assert "flagtree.debug.collect_begin" not in ttir
    assert "flagtree.debug.collect_end" not in ttir
    assert out.metadata.debug_enabled is True


@pytest.mark.module_a
@pytest.mark.module_a_a1
def test_module_a_A1_no_collect_debug_disabled(fresh_triton_cache):
    _ = fresh_triton_cache

    @triton.jit
    def kernel(x_ptr):
        tl.store(x_ptr + tl.arange(0, 4), tl.zeros([4], dtype=tl.float32))

    src = _ast_source(kernel)
    out = triton.compile(src, target=_ASCEND_TARGET)
    assert out.metadata.debug_enabled is False
