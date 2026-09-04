from types import SimpleNamespace

import pytest

import triton.experimental.tle as tle
import triton.language as tl
from triton import backends as backend_registry
from triton import _flagtree_backend
from triton.compiler.code_generator import CodeGenerator
from triton.language import load as tl_load


@pytest.mark.parametrize(
    "primitive, expected",
    [
        ("tle.cumsum", "cumsum"),
        ("language.gpu.core.alloc", "gpu.alloc"),
        ("triton.experimental.tle.language.raw.core.call", "raw.call"),
        (tle.language.cumsum, "cumsum"),
        *([
            (tle.language.gpu.alloc, "gpu.alloc"),
            (tle.language.gpu.buffered_tensor.slot, "gpu.buffered_tensor.slot"),
            (tle.language.gpu.pipeline, "gpu.pipeline"),
        ] if tle.language.gpu is not None else []),
    ],
)
def test_primitive_name(primitive, expected):
    assert tle.primitive_name(primitive) == expected


def test_non_tle_callable_is_rejected():
    with pytest.raises(ValueError, match="not listed in TLE_PRIMITIVES"):
        tle.primitive_name(len)


def test_backend_extension_is_resolved_from_total(monkeypatch):

    def make_view():
        pass

    make_view.__module__ = "triton.backends.tileir.extend_core"
    monkeypatch.setattr(tl, "ext", SimpleNamespace(make_view=make_view))
    assert tle.primitive_name(make_view) == "make_view"


def test_missing_or_empty_backend_config_is_strict(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: None)
    assert tle.get_supported_primitives("missing") == frozenset()
    assert not tle.is_primitive_supported("missing", "gpu.alloc")

    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    assert not tle.is_primitive_supported("empty", "gpu.alloc")


def test_primitive_outside_total_is_not_checked(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: None)
    for primitive in ("gpu.pipe_value.reader", "ext.make_view"):
        assert primitive not in tle.TLE_PRIMITIVES
        assert tle.is_primitive_supported("missing", primitive)
        tle.require_tle("missing", primitive)


@pytest.mark.skipif(tle.language.gpu is None, reason="tle.gpu is not available on this backend")
def test_backend_config_is_normalized(monkeypatch):
    config = SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=["tle.cumsum", "language.gpu.core.alloc"])
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: config)
    assert tle.get_supported_primitives("test") == frozenset({"cumsum", "gpu.alloc"})
    assert tle.is_primitive_supported("test", tle.language.gpu.alloc)


def test_backend_config_is_loaded_from_dedicated_module(monkeypatch):
    config = SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[])
    compiler = SimpleNamespace(__module__="vendor.backend.compiler")
    monkeypatch.setattr(backend_registry, "backends", {"vendor": SimpleNamespace(compiler=compiler)})
    monkeypatch.setattr(tle.importlib, "import_module", lambda module: config
                        if module == "vendor.backend.tle_supported" else None)
    assert tle._backend_config_module("vendor") is config


@pytest.mark.parametrize("configured", [("gpu.alloc", ), ["gpu.alloc", 1]])
def test_invalid_backend_config_is_rejected(monkeypatch, configured):
    config = SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=configured)
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: config)
    with pytest.raises(TypeError, match="TLE_SUPPORTED_PRIMITIVES"):
        tle.get_supported_primitives("invalid")


@pytest.mark.skipif(tle.language.gpu is None, reason="tle.gpu is not available on this backend")
def test_unsupported_error_names_backend_and_primitive(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'test'.*'gpu\.alloc'"):
        tle.require_tle("test", tle.language.gpu.alloc)


def test_active_backend_name_prefers_flagtree_backend(monkeypatch):
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "hcu")
    assert _flagtree_backend.get_active_backend_name() == "hcu"


@pytest.mark.parametrize("backend_name", ["nvidia", "amd"])
def test_active_backend_name_falls_back_for_default_backends(monkeypatch, backend_name):
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "")
    monkeypatch.setattr(
        backend_registry,
        "backends",
        {
            "nvidia": SimpleNamespace(driver=SimpleNamespace(is_active=lambda: backend_name == "nvidia")),
            "amd": SimpleNamespace(driver=SimpleNamespace(is_active=lambda: backend_name == "amd")),
        },
    )
    assert _flagtree_backend.get_active_backend_name() == backend_name


@pytest.mark.skipif(tle.language.gpu is None, reason="tle.gpu is not available on this backend")
def test_codegen_guard_only_rejects_unsupported_tle_primitives(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    monkeypatch.setattr(_flagtree_backend, "get_active_backend_name", lambda: "test")
    generator = object.__new__(CodeGenerator)

    generator._require_tle_primitive(tl_load)
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'test'.*'gpu\.alloc'"):
        generator._require_tle_primitive(tle.language.gpu.alloc)

    def make_view():
        pass

    make_view.__module__ = "triton.backends.tileir.extend_core"
    monkeypatch.setattr(tl, "ext", SimpleNamespace(make_view=make_view))
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'test'.*'make_view'"):
        generator._require_tle_primitive(make_view)
