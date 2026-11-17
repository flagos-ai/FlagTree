# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host regressions for the Iluvatar TLE copy stride contract."""

from types import SimpleNamespace

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.experimental.tle.language.gpu import core


@pytest.fixture
def sme_copy(monkeypatch):
    monkeypatch.setattr(core.iluvatar_copy, "enabled", lambda: True)
    monkeypatch.setattr(core.mthreads_common, "enabled", lambda: False)
    monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", False)
    calls = []
    builder = SimpleNamespace(
        create_async_copy_global_to_local=lambda *args: calls.append(("ordinary", args)),
        create_async_copy_global_to_local_with_stride=lambda *args: calls.append(("sme", args)),
    )
    semantic = SimpleNamespace(
        builder=builder,
        to_tensor=lambda value: value if isinstance(value, tl.tensor) else tl.tensor(value, tl.int32),
        _str_to_load_cache_modifier=lambda value: value,
        _str_to_eviction_policy=lambda value: value,
    )
    source = tl.tensor("source", tl.block_type(tl.pointer_type(tl.float32), [16]))
    target = core.tle.buffered_tensor("shared", tl.float32, [16], core.tle.smem, semantic=semantic)
    return source, target, semantic, calls


@pytest.mark.parametrize("stride", [64, tl.constexpr(64), tl.tensor("runtime_stride", tl.int32)])
def test_async_copy_preserves_sme_row_stride(sme_copy, stride):
    source, target, semantic, calls = sme_copy
    core.copy(source, target, [16], is_async=True, input_stride=stride, _semantic=semantic)
    assert len(calls) == 1 and calls[0][0] == "sme"
    assert calls[0][1][4] == ("runtime_stride" if isinstance(stride, tl.tensor) else 64)


def test_default_async_copy_retains_ordinary_builder(sme_copy):
    source, target, semantic, calls = sme_copy
    core.copy(source, target, [16], is_async=True, _semantic=semantic)
    assert len(calls) == 1 and calls[0][0] == "ordinary"


@pytest.mark.parametrize("is_async", [False, True])
def test_stride_rejects_other_backends_before_copy(monkeypatch, sme_copy, is_async):
    source, target, semantic, calls = sme_copy
    monkeypatch.setattr(core.iluvatar_copy, "enabled", lambda: False)
    with pytest.raises(RuntimeError, match="Iluvatar SME load path"):
        core.copy(source, target, [16], is_async=is_async, input_stride=64, _semantic=semantic)
    assert calls == []


def test_strided_async_copy_requires_builder_capability(sme_copy):
    source, target, semantic, calls = sme_copy
    del semantic.builder.create_async_copy_global_to_local_with_stride
    with pytest.raises(RuntimeError, match="strided TLE async-copy support"):
        core.copy(source, target, [16], is_async=True, input_stride=64, _semantic=semantic)
    assert calls == []


@pytest.mark.parametrize("common_ir", [False, True])
@pytest.mark.parametrize("iluvatar", [False, True])
def test_descriptor_copy_rejects_stride_before_emission(monkeypatch, sme_copy, common_ir, iluvatar):
    _, target, semantic, calls = sme_copy
    monkeypatch.setattr(core.iluvatar_copy, "enabled", lambda: iluvatar)
    monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", common_ir)
    monkeypatch.setattr(core.tle_semantic, "copy", lambda *args: calls.append(("common_ir", args)))
    semantic._convert_to_ir_values = lambda offsets, require_i64=False: offsets
    semantic.builder.create_tma_copy = lambda *args, **kwargs: calls.append(("tma", args))
    descriptor = object.__new__(tl.tensor_descriptor)
    descriptor.handle = "descriptor"
    descriptor.shape = [16]
    with pytest.raises((ValueError, RuntimeError), match="input_stride"):
        core.copy(descriptor, target, [16], [0], input_stride=64, _semantic=semantic)
    assert calls == []


@pytest.mark.parametrize("is_async", [False, True])
def test_common_ir_copy_rejects_stride_before_emission(monkeypatch, sme_copy, is_async):
    source, target, semantic, calls = sme_copy
    monkeypatch.setattr(core.tle_semantic, "COMMON_IR_ENABLED", True)
    monkeypatch.setattr(core.tle_semantic, "copy", lambda *args: calls.append(("common_ir", args)))
    with pytest.raises(ValueError, match="input_stride"):
        core.copy(source, target, [16], input_stride=64, is_async=is_async, _semantic=semantic)
    assert calls == []


def test_shared_to_global_copy_rejects_stride_before_emission(monkeypatch, sme_copy):
    source, target, semantic, calls = sme_copy
    monkeypatch.setattr(core, "_make_full_indices", lambda *args: source)
    monkeypatch.setattr(core, "local_ptr", lambda *args, **kwargs: source)
    monkeypatch.setattr(core.tl, "load", lambda *args, **kwargs: tl.tensor("value", tl.block_type(tl.float32, [16])))
    semantic.store = lambda *args: calls.append(("store", args))
    with pytest.raises(ValueError, match="input_stride"):
        core.copy(target, source, [16], input_stride=64, _semantic=semantic)
    assert calls == []


def test_constexpr_none_stride_retains_default_dispatch(sme_copy):
    source, target, semantic, calls = sme_copy
    core.copy(source, target, [16], is_async=True, input_stride=tl.constexpr(None), _semantic=semantic)
    assert len(calls) == 1 and calls[0][0] == "ordinary"


@pytest.mark.parametrize("stride", [64, tl.constexpr(64), tl.tensor("runtime_stride", tl.int32)])
def test_synchronous_copy_preserves_sme_row_stride(monkeypatch, sme_copy, stride):
    source, target, semantic, calls = sme_copy
    monkeypatch.setattr(core, "_make_full_indices", lambda *args: source)
    monkeypatch.setattr(core, "local_ptr", lambda *args, **kwargs: source)
    semantic.load = lambda *args: calls.append(("load", args)) or tl.tensor("value", tl.block_type(tl.float32, [16]))
    semantic.store = lambda *args: calls.append(("store", args))
    core.copy(source, target, [16], input_stride=stride, _semantic=semantic)
    assert [name for name, _ in calls] == ["load", "store"]
    forwarded_stride = calls[0][1][3]
    assert forwarded_stride.handle == ("runtime_stride" if isinstance(stride, tl.tensor) else 64)


@triton.jit
def _sme_stride_ir_kernel(source):
    row = tl.arange(0, 16)[:, None]
    col = tl.arange(0, 32)[None, :]
    target = tle.gpu.alloc([16, 32], dtype=tl.float16, scope=tle.gpu.smem)
    core.copy(source + row * 64 + col, target, [16, 32], is_async=True, input_stride=64)
    core.async_commit_group()
    core.async_wait_group(0)


@pytest.mark.skipif(not core.iluvatar_copy.enabled(), reason="requires the Iluvatar compiler")
def test_sme_stride_reaches_native_tle_builder():
    from triton._filecheck import run_parser
    from triton.backends.compiler import GPUTarget
    from triton.runtime.jit import MockTensor

    module = run_parser(_sme_stride_ir_kernel, args=(MockTensor(tl.float16), ), target=GPUTarget("corex", 71, 64))
    text = module.str_nodebug()
    assert "ttg.async_copy_global_to_local" in text and "inputStride" in text
    assert "ttg.async_commit_group" in text and "ttg.async_wait" in text
