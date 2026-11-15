"""TLE must find an installed flagcx package when the wheel bundles none.

A wheel built with USE_FLAGCX=OFF carries no flagcx_wrapper, so the detection in
the backend's distributed.py raised ImportError and TLE distributed support
stayed off even with a flagcx distribution installed. The runtime path lookup
below it already prefers that distribution; the import has to do the same.
"""
import importlib
import sys
import types

import pytest

_EXPORTS = ("FLAGCXLibrary", "flagcxDevCommRequirements", "flagcxUniqueId", "FLAGCX_WIN_COLL_SYMMETRIC")
_BACKENDS = ["nvidia", "iluvatar"]


def _stub(name, library):
    module = types.ModuleType(name)
    for export in _EXPORTS:
        setattr(module, export, library if export == "FLAGCXLibrary" else object)
    return module


@pytest.mark.parametrize("backend", _BACKENDS)
def test_detector_falls_back_to_an_installed_flagcx(monkeypatch, backend):
    detector = pytest.importorskip(f"triton.backends.{backend}.distributed")
    monkeypatch.setitem(sys.modules, f"triton.backends.{backend}.flagcx_wrapper", None)
    monkeypatch.setitem(sys.modules, "flagcx.api", _stub("flagcx.api", object))
    assert detector.flagcx_packages_detector() is True


@pytest.mark.parametrize("backend", _BACKENDS)
def test_detector_without_any_flagcx(monkeypatch, backend):
    detector = pytest.importorskip(f"triton.backends.{backend}.distributed")
    monkeypatch.setitem(sys.modules, f"triton.backends.{backend}.flagcx_wrapper", None)
    monkeypatch.setitem(sys.modules, "flagcx.api", None)
    with pytest.raises(ImportError):
        detector.flagcx_packages_detector()


def test_communication_loads_the_installed_library(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    so_file = tmp_path / "libflagcx.so"
    so_file.write_text("not a shared object\n")

    class _Library:

        def __init__(self, so_file=None):
            self.so_file = so_file

    conf = types.SimpleNamespace(is_available=True, shared_lib_path=so_file, include_path=tmp_path)
    backend = types.ModuleType("triton.backends.nvidia.distributed")
    backend.flagcx_rt_conf = conf

    monkeypatch.setitem(sys.modules, "triton.backends.nvidia.distributed", backend)
    monkeypatch.setitem(sys.modules, "triton.experimental.tle.language.flagcx_wrapper", None)
    monkeypatch.setitem(sys.modules, "flagcx.api", _stub("flagcx.api", _Library))
    # communication resolves flagcx_rt_conf by backend name at import time,
    # and the built wheel's marker says "amd", which ships no distributed.py;
    # point it at the backend this test stubs instead.
    monkeypatch.setattr("triton._flagtree_backend.FLAGTREE_BACKEND", "nvidia")
    monkeypatch.delitem(sys.modules, "triton.experimental.tle.language.communication", raising=False)

    communication = importlib.import_module("triton.experimental.tle.language.communication")
    assert str(communication.flagcx.so_file) == str(so_file)
