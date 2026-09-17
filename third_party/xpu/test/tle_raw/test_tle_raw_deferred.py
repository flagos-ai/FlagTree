"""Tests for deferred `tle.raw` payloads on the XPU cluster path.

A deferred payload records only a content-addressed id at trace time
(`triton_xpu.raw_source_id`, empty `llvm_ir`); the backend compiles it in
`make_llir` for the arch it is building for and
`tritonxpu-materialize-deferred-raw` fills the payload in.

This file covers the frontend halves that need no device and no toolchain:
registry / arch resolution / toolchain lookup, the cache key, the pending-source
store, and the backend hook (with a stub handle). The IR halves -- builder,
materialization pass, lowering -- live in test_tle_raw_cluster_ir.py.
"""

import os

import pytest

from triton.experimental.tle.raw import registry
from triton.experimental.tle.raw.deferred import materialize_deferred_raw
from triton.experimental.tle.raw.runtime import XPUJITFunction
from triton.experimental.tle.raw.source_store import (
    clear_pending_sources,
    list_pending_sources,
)

pytestmark = pytest.mark.no_xpu_required

PAYLOAD = ("llvm.func @my_scale(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, "
           "%arg2: i32) { llvm.return }")


@pytest.fixture(autouse=True)
def _clean_source_store():
    clear_pending_sources()
    yield
    clear_pending_sources()


def _handle(tmp_path, dialect="xpu3", source='extern "C" __device__ void my_scale() {}', **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "my_scale.xpu"
    path.write_text(source)
    cls = registry[dialect]

    def my_scale(out, inp):
        ...

    return cls(my_scale, file=path, **kwargs)


# -- frontend: dialects, arch and toolchain resolution ----------------------


def test_registry_covers_the_xpu_stack_only():
    """FlagTree ships one stack; the internal mars/jupiter names are not mapped."""
    assert registry["xpu3"] is XPUJITFunction
    assert set(registry) == {"xpu3"}


def test_unknown_dialect_is_rejected():
    import triton.experimental.tle as tle

    with pytest.raises(ValueError, match="unknown tle.raw dialect"):
        tle.raw.dialect("xpu5")


def test_arch_resolution_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_XPU_ARCH", raising=False)
    assert _handle(tmp_path, "xpu3").resolve_arch() == 3
    # Env var beats the default, an explicit arch= beats the env var, and the
    # caller (the backend, during materialization) beats both.
    monkeypatch.setenv("TRITON_XPU_ARCH", "4")
    assert _handle(tmp_path, "xpu3").resolve_arch() == 4
    assert _handle(tmp_path, "xpu3", arch=3).resolve_arch() == 3
    assert _handle(tmp_path, "xpu3", arch=3).resolve_arch(5) == 5


def test_toolchain_dir_is_the_packaged_per_arch_one(tmp_path):
    """XPUBackend keys the clang dir off `arch`; handing it the wrong option
    object would raise AttributeError instead."""
    if "TRITON_XPU_CLANG_PATH" in os.environ:
        pytest.skip("TRITON_XPU_CLANG_PATH overrides the packaged toolchain layout")
    clang_dir = _handle(tmp_path, "xpu3")._backend_clang_dir(3)
    assert clang_dir.name == "bin" and "xpu3" in str(clang_dir)


def test_a_payload_without_file_or_source_is_rejected():

    def my_scale():
        ...

    with pytest.raises(ValueError, match="needs either `source` or `file`"):
        XPUJITFunction(my_scale)


# -- cache key --------------------------------------------------------------


def test_editing_the_payload_changes_the_cache_key(tmp_path):
    handle = _handle(tmp_path, "xpu3")
    before = handle.cache_key
    handle.file.write_text('extern "C" __device__ void my_scale() { /* v2 */ }')
    assert handle.cache_key != before


def test_dependencies_finder_folds_the_payload_in(tmp_path):
    from triton.runtime.jit import DependenciesFinder

    handle = _handle(tmp_path, "xpu3")
    finder = DependenciesFinder("kernel", {}, {}, "def kernel():\n    pass\n")
    baseline = finder.ret
    finder.record_reference(handle, {}, "my_scale")
    assert finder.ret != baseline
    # Hashed, not tracked as a mutable global (which would deepcopy the handle).
    assert not finder.used_global_vals


# -- source store -----------------------------------------------------------


def test_pending_sources_are_content_addressed(tmp_path):
    first = _handle(tmp_path, "xpu3").register_pending_source()
    second = _handle(tmp_path, "xpu3").register_pending_source()
    assert first == second
    assert list(list_pending_sources()) == [first]
    assert list_pending_sources()[first]["dialect"] == "xpu"


def test_pending_id_tracks_the_source(tmp_path):
    original = _handle(tmp_path / "a", "xpu3").register_pending_source()
    edited = _handle(tmp_path / "b", "xpu3", source='extern "C" __device__ void my_scale() { }')
    assert original != edited.register_pending_source()


# -- backend hook -----------------------------------------------------------


class _StubHandle:
    """Payload handle that needs no toolchain; records the arch it got."""

    def __init__(self):
        self.arches = []

    def make_llvm(self, context=None, arch=None):
        self.arches.append(arch)
        return PAYLOAD


def test_materialize_hook_compiles_for_the_backend_arch():
    from triton.experimental.tle.raw.source_store import register_source

    stub = _StubHandle()
    sid = register_source(dialect="xpu", callee="my_scale", source="src", handle=stub)

    added = []
    compiled = materialize_deferred_raw(None, lambda pm, sources: added.append(sources), arch=3)

    assert stub.arches == [3]
    assert compiled == {sid: PAYLOAD}
    assert added == [compiled]


def test_materialize_hook_is_a_noop_without_pending_payloads():
    """Which is what every compile-cache hit looks like: tracing never runs, so
    nothing is pending and no pass is added."""
    added = []
    assert materialize_deferred_raw(None, lambda pm, sources: added.append(sources), arch=3) == {}
    assert not added


def test_materialize_hook_can_filter_by_dialect():
    from triton.experimental.tle.raw.source_store import register_source

    register_source(dialect="xpu", callee="my_scale", source="src", handle=_StubHandle())
    added = []
    assert materialize_deferred_raw(None, lambda pm, sources: added.append(sources), arch=3, dialect="other") == {}
    assert not added


# -- the SDNN path is not available here ------------------------------------


def test_sdnn_path_is_rejected_with_an_explanation():
    """`sdnn.raw` needs TritonSDNN's IR headers, which FlagTree does not ship."""
    from triton.experimental.tle.language.raw.core import _SDNN_MESSAGE

    assert "is_sdnn" in _SDNN_MESSAGE
    assert "TritonSDNN" in _SDNN_MESSAGE
