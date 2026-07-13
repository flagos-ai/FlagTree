# SPDX-License-Identifier: MIT
"""A-2: ``CompiledKernel.metadata`` 与契约 §3.1 / §2.2 / README 对齐。

契约 ID: **A-2**；依赖 ``triton.compile``（CUDA target + torch）。
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
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

# Minimum keys module D / TrackedOpTable JSON 消费路径应能依赖（与 TrackedOpEntry 字段对应，camelCase JSON）。
_TRACKED_OP_ENTRY_REQUIRED_KEYS = frozenset({
    "opId",
    "scopeId",
    "resultIndex",
    "isMemoryOp",
    "opCategory",
    "role",
    "mlirOpName",
    "sourceLoc",
    "tritonStatement",
    "inlineCallPath",
    "result",
    "operands",
    "addrSpace",
    "accessType",
    "accessBytes",
    "alignmentRequired",
    "hasMask",
    "maskDtype",
    "cacheModifier",
    "evictionPolicy",
    "isVolatile",
    "boundaryCheckPolicy",
    "paddingSemantics",
})

_RESULT_VALUE_REQUIRED_KEYS = frozenset({
    "valueKind",
    "dtype",
    "elementDtype",
    "shape",
    "stride",
    "layout",
    "encoding",
    "addrSpace",
    "rank",
    "elementBits",
    "vecWidth",
})


def _kernel_collect():

    @triton.jit
    def kernel(x_ptr):
        offsets = tl.arange(0, 4)
        tl.debug_collect_start(level=1)
        x = tl.load(x_ptr + offsets)
        tl.store(x_ptr + offsets, x)
        tl.debug_collect_end()

    return kernel


def _source(fn):
    ast_params = inspect.signature(ASTSource).parameters
    if "constexprs" in ast_params:
        return ASTSource(fn=fn, signature={"x_ptr": "*fp32"}, constexprs={})
    return ASTSource(fn=fn, signature={"x_ptr": "*fp32"}, constants={})


def _target():
    if os.environ.get("FLAGTREE_BACKEND") == "ascend":
        return GPUTarget("npu", os.environ.get("ASCEND_TEST_ARCH", "Ascend910B"), 0)
    return GPUTarget("cuda", 80, 32)


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_metadata_keys_when_collect(fresh_triton_cache, monkeypatch):
    monkeypatch.delenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", raising=False)
    _ = fresh_triton_cache
    kernel = _kernel_collect()
    src = _source(kernel)
    out = triton.compile(src, target=_target())
    md = out.metadata
    assert md.debug_enabled is True
    assert md.debug_protocol_version == 2
    assert md.debug_record_level == 1
    assert md.debug_addr_level == 0
    assert md.debug_export_mode == "POST_KERNEL_EXPORT"
    assert isinstance(md.debug_kernel_id, int)
    assert md.debug_kernel_id != 0
    assert isinstance(md.debug_tracked_table, list)
    assert len(md.debug_tracked_table) > 0
    row = md.debug_tracked_table[0]
    assert isinstance(row, dict)
    assert row["sourceLoc"]
    missing = _TRACKED_OP_ENTRY_REQUIRED_KEYS - row.keys()
    assert not missing, f"tracked row missing keys {missing}"
    res = row["result"]
    assert isinstance(res, dict)
    assert not (_RESULT_VALUE_REQUIRED_KEYS - res.keys())
    assert md.debug_launch_hidden_arg is False


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_debug_launch_hidden_arg_follows_env(fresh_triton_cache, monkeypatch):
    monkeypatch.setenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "1")
    _ = fresh_triton_cache
    kernel = _kernel_collect()
    src = _source(kernel)
    out = triton.compile(src, target=_target())
    assert out.metadata.debug_launch_hidden_arg is True


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_zero_record_collect_disables_hidden_arg(fresh_triton_cache, monkeypatch):
    monkeypatch.setenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", "1")
    _ = fresh_triton_cache

    @triton.jit
    def kernel(x_ptr):
        tl.debug_collect_start(level=1)
        offsets = tl.arange(0, 4)
        y = offsets + 1
        tl.debug_collect_end()
        tl.store(x_ptr + offsets, y.to(tl.float32))

    src = _source(kernel)
    out = triton.compile(src, target=_target())
    assert out.metadata.debug_enabled is True
    assert out.metadata.debug_records_per_instance == 0
    assert out.metadata.debug_launch_hidden_arg is False


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_debug_launch_hidden_arg_follows_debugger_api(monkeypatch):
    from triton.compiler import flagtree_debug

    monkeypatch.delenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", raising=False)
    triton.enable_debug()
    try:
        assert flagtree_debug._debug_launch_hidden_arg_enabled() is True
    finally:
        triton.disable_debug()


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_metadata_namedtuple_dict_and_json(fresh_triton_cache, monkeypatch):
    monkeypatch.delenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", raising=False)
    _ = fresh_triton_cache
    kernel = _kernel_collect()
    src = _source(kernel)
    out = triton.compile(src, target=_target())
    md = out.metadata
    d = md._asdict()
    for key in (
            "debug_enabled",
            "debug_protocol_version",
            "debug_record_level",
            "debug_addr_level",
            "debug_export_mode",
            "debug_kernel_id",
            "debug_tracked_table",
            "debug_launch_hidden_arg",
    ):
        assert key in d
    payload = {
        "debug_kernel_id": md.debug_kernel_id,
        "debug_tracked_table": md.debug_tracked_table,
    }
    text = json.dumps(payload)
    assert "debug_kernel_id" in text
    for row in md.debug_tracked_table:
        json.dumps(row)


@pytest.mark.module_a
@pytest.mark.module_a_a2
def test_module_a_A2_no_collect_debug_disabled(fresh_triton_cache, monkeypatch):
    monkeypatch.delenv("TRITON_FLAGTREE_DEBUG_LAUNCH_PTR", raising=False)
    _ = fresh_triton_cache

    @triton.jit
    def kernel(x_ptr):
        tl.store(x_ptr + tl.arange(0, 4), tl.zeros([4], dtype=tl.float32))

    src = _source(kernel)
    out = triton.compile(src, target=_target())
    assert out.metadata.debug_enabled is False
    assert out.metadata.debug_launch_hidden_arg is False
