"""An unloadable libflagcx.so must surface as ImportError, not OSError.

Regression test for https://github.com/flagos-ai/FlagTree/issues/1234. The wheel
bundles a libflagcx.so linked against libcudart.so.12, so on a CUDA 13 machine
the eager dlopen in tle.language.communication raised OSError out of ``import
triton.experimental.tle``. The TLE probes around it -- flag_gems'
``has_triton_tle``, python/test/unit/runtime/test_launch.py,
triton/compiler/code_generator.py -- guard with ``except ImportError``, so the
OSError escaped them and ``import flag_gems`` failed as a whole.
"""
import importlib
import sys
import types

import pytest


class _UnloadableFlagcxLibrary:

    def __init__(self, so_file):
        raise OSError("libcudart.so.12: cannot open shared object file: No such file or directory")


def test_unloadable_flagcx_raises_import_error(tmp_path, monkeypatch):
    # the FlagCX packages are installed -- the library is there and the nvidia
    # backend reports them as available -- but the library cannot be loaded
    so_file = tmp_path / "libflagcx.so"
    so_file.write_text("not a shared object\n")

    conf = types.SimpleNamespace(is_available=True, shared_lib_path=so_file, include_path=tmp_path)
    distributed = types.ModuleType("triton.backends.nvidia.distributed")
    distributed.flagcx_rt_conf = conf

    wrapper = types.ModuleType("triton.experimental.tle.language.flagcx_wrapper")
    wrapper.FLAGCXLibrary = _UnloadableFlagcxLibrary
    wrapper.flagcxDevCommRequirements = object
    wrapper.flagcxUniqueId = object
    wrapper.FLAGCX_WIN_COLL_SYMMETRIC = 1

    monkeypatch.setitem(sys.modules, "triton.backends.nvidia.distributed", distributed)
    monkeypatch.setitem(sys.modules, "triton.experimental.tle.language.flagcx_wrapper", wrapper)
    monkeypatch.delitem(sys.modules, "triton.experimental.tle.language.communication", raising=False)

    # tle.language imports communication and tle imports language, so this is the
    # error plain `import triton.experimental.tle` reports
    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("triton.experimental.tle.language.communication")

    assert "libflagcx" in str(excinfo.value)
    # the dlopen error is preserved, so the cause stays diagnosable
    assert isinstance(excinfo.value.__cause__, OSError)
