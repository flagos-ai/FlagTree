from types import SimpleNamespace

import pytest

import triton.experimental.tle as tle
import triton.language as tl
from triton import _flagtree_backend
from triton import backends as backend_registry
from triton.compiler.code_generator import CodeGenerator
from triton.language import load as tl_load


@pytest.mark.parametrize(
    "primitive, expected",
    [
        ("tle.cumsum", "cumsum"),
        ("language.gpu.core.alloc", "gpu.alloc"),
        ("triton.experimental.tle.language.raw.core.call", "raw.call"),
        (tle.language.cumsum, "cumsum"),
        (tle.language.gpu.alloc, "gpu.alloc"),
        (tle.language.gpu.buffered_tensor.load, "gpu.buffered_tensor.load"),
        (tle.language.gpu.buffered_tensor.store, "gpu.buffered_tensor.store"),
        (tle.language.gpu.buffered_tensor.slot, "gpu.buffered_tensor.slot"),
        (tle.language.gpu.buffered_tensor.subslice, "gpu.buffered_tensor.subslice"),
        (tle.language.gpu.pipeline, "gpu.pipeline"),
    ],
)
def test_primitive_name(primitive, expected):
    assert tle.primitive_name(primitive) == expected


def test_non_tle_callable_is_rejected():
    with pytest.raises(ValueError, match="not listed in TLE_PRIMITIVES"):
        tle.primitive_name(len)


def test_buffer_access_uses_methods():
    for name in ("to_tensor", "store_tensor"):
        assert not hasattr(tle.language.gpu, name)
        assert f"gpu.{name}" not in tle.TLE_PRIMITIVES
    for name in ("load", "store"):
        primitive = getattr(tle.language.gpu.buffered_tensor, name)
        assert tle.primitive_name(primitive) == f"gpu.buffered_tensor.{name}"


def test_backend_extension_is_resolved_from_total(monkeypatch):

    def make_view():
        pass

    make_view.__module__ = "triton.backends.tileir.extend_core"
    monkeypatch.setattr(tl, "ext", SimpleNamespace(make_view=make_view))
    assert tle.primitive_name(make_view) == "make_view"


def test_missing_or_empty_backend_config_is_strict(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: None)
    assert tle.get_supported_primitives() == frozenset()
    assert not tle.is_tle_supported("gpu.alloc")

    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    assert not tle.is_tle_supported("gpu.alloc")


def test_primitive_outside_total_is_not_checked(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: None)
    for primitive in ("gpu.pipe_value.reader", "ext.make_view"):
        assert primitive not in tle.TLE_PRIMITIVES
        assert tle.is_tle_supported(primitive)
        tle.require_tle(primitive)


def test_backend_config_is_normalized(monkeypatch):
    config = SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=["tle.cumsum", "language.gpu.core.alloc"])
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: config)
    assert tle.get_supported_primitives() == frozenset({"cumsum", "gpu.alloc"})
    assert tle.is_tle_supported(tle.language.gpu.alloc)


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
        tle.get_supported_primitives()


def test_unsupported_error_names_backend_and_primitive(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "test")
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'test'.*'gpu\.alloc'"):
        tle.require_tle(tle.language.gpu.alloc)


def test_tileir_build_checks_whitelist_of_routed_backend(monkeypatch):
    # TileIR builds leave FLAGTREE_BACKEND empty. Like the router, a primitive
    # set goes to TileIR only with FLAGTREE_USE_TILEIR=1 and when TileIR's
    # whitelist covers it; otherwise it is checked against NVIDIA's whitelist.
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "")
    whitelists = {
        "nvidia": SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=["gpu.wgmma"]),
        "tileir": SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=["make_view"]),
    }
    monkeypatch.setattr(tle, "_backend_config_module", whitelists.get)
    monkeypatch.setattr(backend_registry, "backends", {"nvidia": object(), "tileir": object()})
    t, n = ["make_view"], ["gpu.wgmma"]

    monkeypatch.setenv("FLAGTREE_USE_TILEIR", "1")
    assert tle.get_supported_primitives() == frozenset({"gpu.wgmma", "make_view"})
    assert tle.is_tle_supported(t)
    assert tle.is_tle_supported(n)
    assert not tle.is_tle_supported(t + n)
    tle.require_tle(t)
    tle.require_tle(n)
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'nvidia'.*'make_view'"):
        tle.require_tle(t + n)

    monkeypatch.delenv("FLAGTREE_USE_TILEIR")
    assert tle.get_supported_primitives() == frozenset({"gpu.wgmma"})
    assert not tle.is_tle_supported(t)
    assert tle.is_tle_supported(n)
    assert not tle.is_tle_supported(t + n)
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'nvidia'.*'make_view'"):
        tle.require_tle(t)
    tle.require_tle(n)
    with pytest.raises(tle.TLEUnsupportedPrimitiveError, match=r"backend 'nvidia'.*'make_view'"):
        tle.require_tle(t + n)


@pytest.mark.parametrize("backend_name", ["hcu", "amd", "mthreads", "iluvatar"])
def test_tle_logical_descriptor_frontend_gate_rejects_other_backends(monkeypatch, backend_name):
    from triton.experimental.tle.language.gpu.semantic import TLEFrontendSemantic

    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", backend_name)
    semantic = object.__new__(TLEFrontendSemantic)
    with pytest.raises(ValueError, match=r"TLE logical descriptors.*NVIDIA backend"):
        semantic.make_tensor_descriptor(None, [], [], [80, 256])


def test_tle_power_of_two_descriptor_is_forwarded(monkeypatch):
    from triton.experimental.tle.language.gpu.semantic import TLEFrontendSemantic
    from triton.language.semantic import TritonSemantic

    forwarded = object()
    monkeypatch.setattr(TritonSemantic, "make_tensor_descriptor", lambda self, *args: forwarded)
    semantic = object.__new__(TLEFrontendSemantic)
    assert semantic.make_tensor_descriptor(None, [], [], [128, 256]) is forwarded


def test_codegen_guard_only_rejects_unsupported_tle_primitives(monkeypatch):
    monkeypatch.setattr(tle, "_backend_config_module", lambda _backend: SimpleNamespace(TLE_SUPPORTED_PRIMITIVES=[]))
    monkeypatch.setattr(_flagtree_backend, "FLAGTREE_BACKEND", "test")
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
